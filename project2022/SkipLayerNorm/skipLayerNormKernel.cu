 /*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#if CUDA_VERSION >= 10010

#include "NvInfer.h"
#include "bertCommon.h"
#include "common.cuh"
#include "serialize.hpp"
#include "skipLayerNormPlugin.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

 

 
 namespace bert
 {
 
 // Clip plugin specific constants
 namespace
 {
 const char* SKIP_LAYER_NORM_VERSION{"1"};
 const char* SKIP_LAYER_NORM_NAME{"MySkipLNPluginDynamic"};
 const char* SKIP_LAYER_NORM_VAR_SEQLEN_VERSION{"2"};
 } // namespace
 
 // Static class fields initialization
 PluginFieldCollection SkipLayerNormPluginDynamicCreator::mFC{};
 std::vector<PluginField> SkipLayerNormPluginDynamicCreator::mPluginAttributes;
 
 PluginFieldCollection SkipLayerNormVarSeqlenPluginCreator::mFC{};
 std::vector<PluginField> SkipLayerNormVarSeqlenPluginCreator::mPluginAttributes;
 
 REGISTER_TENSORRT_PLUGIN(SkipLayerNormPluginDynamicCreator);
 REGISTER_TENSORRT_PLUGIN(SkipLayerNormVarSeqlenPluginCreator);
 
 static inline DataType getParamWordType(DataType cfgType) noexcept
 {
     if (cfgType == DataType::kINT8)
     {
         return DataType::kHALF;
     }
 
     return cfgType;
 }
 
 SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const DataType type, const int ld,
     const Weights& beta, const Weights& gamma, const Weights& bias)
     : mLayerName(name)
     , mGammaDev(nullptr)
     , mBetaDev(nullptr)
     , mLd(ld)
     , mType(type)
     , mBiasDev(nullptr)
 {
     assert(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kINT8);
     // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
     // mType is the plugin IO datatype, can be int8
     mCfgType = mType == DataType::kINT8 ? DataType::kHALF :  mType;
     mParamWordsize = getElementSize(mCfgType);
 
     mBeta.convertAndCopy(beta, mCfgType);
     mGamma.convertAndCopy(gamma, mCfgType);
 
     mHasBias = (bias.values != nullptr);
     if (mHasBias)
     {
         mBias.convertAndCopy(bias, mCfgType);
     }
 }
 
 SkipLayerNormPluginDynamic::SkipLayerNormPluginDynamic(const std::string name, const void* data, size_t length)
     : mLayerName(name)
     , mGammaDev(nullptr)
     , mBetaDev(nullptr)
     , mBiasDev(nullptr)
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic deserialize");
 
     // Deserialize in the same order as serialization
     deserialize_value(&data, &length, &mType);
     deserialize_value(&data, &length, &mCfgType);
     deserialize_value(&data, &length, &mLd);
     deserialize_value(&data, &length, &mHasBias);
 
     assert(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
     mParamWordsize = getElementSize(mCfgType);
 
     const char* d = static_cast<const char*>(data);
     mBeta.convertAndCopy(d, mLd, mCfgType);
     mGamma.convertAndCopy(d, mLd, mCfgType);
     if (mHasBias)
     {
         mBias.convertAndCopy(d, mLd, mCfgType);
     }
 }
 
 // IPluginV2DynamicExt Methods
 IPluginV2DynamicExt* SkipLayerNormPluginDynamic::clone() const noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic clone");
 
     auto* p = new SkipLayerNormPluginDynamic(mLayerName, mType, mLd, mBeta, mGamma, mBias);
     p->initialize();
     p->setPluginNamespace(mNamespace.c_str());
     return p;
 }
 
 DimsExprs SkipLayerNormPluginDynamic::getOutputDimensions(
     int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
 {
     //assert(nbInputs == 2);
     assert(nbInputs == 1);
     assert(outputIndex == 0);
     // assert(inputs[0].nbDims == inputs[1].nbDims);
     return inputs[0];
 }
 
 bool SkipLayerNormPluginDynamic::supportsFormatCombination(
     int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
 {
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     assert(nbOutputs == 1);
 
     const PluginTensorDesc& in = inOut[pos];
     if (pos == 0)
     {
         // Since H = W = 1, we can report CHWx for any x
         if (mType == DataType::kINT8)
         {
             // won't work for hiddensize too small!
             TensorFormat myFmt = TensorFormat::kCHW32;
             if (mLd < 32)
             {
                 myFmt = TensorFormat::kCHW4;
                 BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW4 for LD=", mLd);
             }
             else
             {
                 BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW32 for LD=", mLd);
             }
             // TODO do we need to check if the vectorization divides mLd?
             return ((in.type == mType) && (in.format == myFmt));
         }
         return (in.type == mType) && (in.format == TensorFormat::kLINEAR);
     }
     const PluginTensorDesc& prev = inOut[pos - 1];
 
     return in.type == prev.type && in.format == prev.format;
 }
 
 void SkipLayerNormPluginDynamic::configurePlugin(
     const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic configurePlugin");
 
     // Validate input arguments
     assert(nbOutputs == 1);
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     if (mType == DataType::kFLOAT || mType == DataType::kHALF)
     {
         assert(mType == inputs[0].desc.type);
         // assert(mType == inputs[1].desc.type);
     }
     else
     {
         assert(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
         // assert(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
     }
     const auto& inDims0 = inputs[0].desc.dims;
     // const auto& inDims1 = inputs[1].desc.dims;
     // TRT_UNUSED inDims1;
     // assert(inDims0.nbDims == inDims1.nbDims);
 
     // assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));
 
     assert(inDims0.nbDims == 5);
     mLd = inDims0.d[HDIM]; // hiddensize
     assert(inDims0.d[3] == 1);
     assert(inDims0.d[4] == 1);
 
     mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;
 
     const auto paramType = getParamWordType(mCfgType);
     mParamWordsize = getElementSize(paramType);
 
     copyToDevice(mGamma, getWeightsSize(mGamma, paramType), mGammaDev);
     copyToDevice(mBeta, getWeightsSize(mBeta, paramType), mBetaDev);
     if (mHasBias)
     {
         copyToDevice(mBias, getWeightsSize(mBias, paramType), mBiasDev);
     }
 }
 
 size_t SkipLayerNormPluginDynamic::getWorkspaceSize(
     const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
 {
     return 0;
 }
 
 int SkipLayerNormPluginDynamic::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
     const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
 {
     const int inputVolume = volume(inputDesc[0].dims);
     int status = -1;
     DataType iType = inputDesc->type;
 
     // Our plugin outputs only one tensor
     // Launch CUDA kernel wrapper and save its return value
     if (iType == DataType::kFLOAT)
     {
         const auto* const input = static_cast<const float*>(inputs[0]);
         // const auto* const skip = static_cast<const float*>(inputs[1]);
         auto* output = static_cast<float*>(outputs[0]);
         const auto* const bias = static_cast<const float*>(mBiasDev.get());
         const auto* const beta = static_cast<const float*>(mBetaDev.get());
         const auto* const gamma = static_cast<const float*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNorm<float, true>(
                 stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias);
         }
         else
         {
             status
                 = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input,  beta, gamma, output, bias);
         }
     }
     else if (iType == DataType::kHALF)
     {
         const auto* const input = static_cast<const half*>(inputs[0]);
         // const auto* const skip = static_cast<const half*>(inputs[1]);
         auto* output = static_cast<half*>(outputs[0]);
         const auto* const bias = static_cast<const half*>(mBiasDev.get());
         const auto* const beta = static_cast<const half*>(mBetaDev.get());
         const auto* const gamma = static_cast<const half*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNorm<half, true>(
                 stream, static_cast<int>(mLd), inputVolume, input,  beta, gamma, output, bias);
         }
         else
         {
             status
                 = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input,  beta, gamma, output, bias);
         }
     }
     else if (iType == DataType::kINT8)
     {
         const float dqScaleIn = inputDesc[0].scale;
         // const float dqScaleSkip = inputDesc[1].scale;
         const float qScale = 1.F / outputDesc[0].scale;
         const auto* const input = static_cast<const int8_t*>(inputs[0]);
         // const auto* const skip = static_cast<const int8_t*>(inputs[1]);
         auto* output = static_cast<int8_t*>(outputs[0]);
         const auto* const bias = static_cast<const half*>(mBiasDev.get());
         const auto* const beta = static_cast<const half*>(mBetaDev.get());
         const auto* const gamma = static_cast<const half*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, beta, gamma,
                 output, bias, dqScaleIn, qScale);
         }
         else
         {
             status = computeSkipLayerNormDQQ<false>(
                 stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias, dqScaleIn, qScale);
         }
     }
     else
     {
         std::cerr << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received " << static_cast<int>(iType) << "." << std::endl;
         assert(false);
     }
     return status;
 }
 
 // IPluginV2Ext Methods
 DataType SkipLayerNormPluginDynamic::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
 {
     assert(index == 0);
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     return inputTypes[0];
 }
 
 // IPluginV2 Methods
 const char* SkipLayerNormPluginDynamic::getPluginType() const noexcept
 {
     return SKIP_LAYER_NORM_NAME;
 }
 
 const char* SkipLayerNormPluginDynamic::getPluginVersion() const noexcept
 {
     return SKIP_LAYER_NORM_VERSION;
 }
 
 int SkipLayerNormPluginDynamic::getNbOutputs() const noexcept
 {
     return 1;
 }
 int SkipLayerNormPluginDynamic::initialize() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic initialize");
     return 0;
 }
 
 void SkipLayerNormPluginDynamic::terminate() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic terminate");
 }
 
 size_t SkipLayerNormPluginDynamic::getSerializationSize() const noexcept
 {
     const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
     return 2 * mParamWordsize * mLd + 2 * sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
 }
 
 void SkipLayerNormPluginDynamic::serialize(void* buffer) const noexcept
 {
     serialize_value(&buffer, mType);
     serialize_value(&buffer, mCfgType);
     serialize_value(&buffer, mLd);
     serialize_value(&buffer, mHasBias);
 
     char* d = static_cast<char*>(buffer);
     serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
     serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
     if (mHasBias)
     {
         serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
     }
 }
 
 void SkipLayerNormPluginDynamic::destroy() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormPluginDynamic destroy");
     // This gets called when the network containing plugin is destroyed
     mGammaDev.reset(nullptr);
     mBetaDev.reset(nullptr);
     mBiasDev.reset(nullptr);
     delete this;
 }
 
 void SkipLayerNormPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept
 {
     mNamespace = libNamespace;
 }
 
 const char* SkipLayerNormPluginDynamic::getPluginNamespace() const noexcept
 {
     return mNamespace.c_str();
 }
 
 /////////////////////////////////////////////////////////
 
 SkipLayerNormPluginDynamicCreator::SkipLayerNormPluginDynamicCreator()
 {
     mFC.nbFields = mPluginAttributes.size();
     mFC.fields = mPluginAttributes.data();
 }
 
 const char* SkipLayerNormPluginDynamicCreator::getPluginName() const noexcept
 {
     return SKIP_LAYER_NORM_NAME;
 }
 
 const char* SkipLayerNormPluginDynamicCreator::getPluginVersion() const noexcept
 {
     return SKIP_LAYER_NORM_VERSION;
 }
 
 const PluginFieldCollection* SkipLayerNormPluginDynamicCreator::getFieldNames() noexcept
 {
     return &mFC;
 }
 
 IPluginV2* SkipLayerNormPluginDynamicCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
 {
     try
     {
         BERT_DEBUG_MSG("SkipLayerNormPluginDynamicCreator createPlugin");
 
         int ld = 0;
         Weights beta{DataType::kFLOAT, nullptr, 0};
         Weights gamma{DataType::kFLOAT, nullptr, 0};
         Weights bias{DataType::kFLOAT, nullptr, 0};
         int typeId = -1;
 
         for (int i = 0; i < fc->nbFields; i++)
         {
             std::string field_name(fc->fields[i].name);
             if (field_name.compare("ld") == 0)
             {
                 ld = *static_cast<const int*>(fc->fields[i].data);
                 BERT_DEBUG_VALUE("Building ld: ", ld);
             }
 
             if (field_name.compare("type_id") == 0)
             {
                 typeId = *static_cast<const int*>(fc->fields[i].data);
                 BERT_DEBUG_VALUE("Building typeId: ", typeId);
             }
 
             if (field_name.compare("beta") == 0)
             {
                 BERT_DEBUG_MSG("Building beta...");
                 beta.values = fc->fields[i].data;
                 beta.count = fc->fields[i].length;
                 beta.type = fieldTypeToDataType(fc->fields[i].type);
             }
 
             if (field_name.compare("gamma") == 0)
             {
                 BERT_DEBUG_MSG("Building gamma...");
                 gamma.values = fc->fields[i].data;
                 gamma.count = fc->fields[i].length;
                 gamma.type = fieldTypeToDataType(fc->fields[i].type);
             }
 
             if (field_name.compare("bias") == 0)
             {
                 BERT_DEBUG_MSG("Building bias...");
                 bias.values = fc->fields[i].data;
                 bias.count = fc->fields[i].length;
                 bias.type = fieldTypeToDataType(fc->fields[i].type);
             }
         }
         BERT_DEBUG_VALUE("fc->nbFields ",fc->nbFields);
         
         BERT_DEBUG_VALUE("Type ", typeId);
 
         if (typeId < 0 || typeId > 3)
         {
             std::cerr << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
             std::cerr << "fc->nbFields no varlen: " << fc->nbFields<< std::endl;
         }
 
         if (beta.count <= 0 || beta.values == nullptr)
         {
            std::cerr << "SkipLayerNorm: invalid beta" << std::endl;
         }
 
         if (gamma.count <= 0 || gamma.values == nullptr)
         {
            std::cerr << "SkipLayerNorm: invalid gamma" << std::endl;
         }
 
         return new SkipLayerNormPluginDynamic(name, static_cast<DataType>(typeId), ld, beta, gamma, bias);
     }
     catch (const std::exception& e)
     {
         caughtError(e);
     }
     return nullptr;
 }
 
 IPluginV2* SkipLayerNormPluginDynamicCreator::deserializePlugin(
     const char* name, const void* serialData, size_t serialLength) noexcept
 {
     // This object will be deleted when the network is destroyed, which will
     // call SkipLayerNormPluginDynamic::destroy()
     try
     {
         return new SkipLayerNormPluginDynamic(name, serialData, serialLength);
     }
     catch (const std::exception& e)
     {
         caughtError(e);
     }
     return nullptr;
 }
 
 void SkipLayerNormPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept
 {
     mNamespace = libNamespace;
 }
 
 const char* SkipLayerNormPluginDynamicCreator::getPluginNamespace() const noexcept
 {
     return mNamespace.c_str();
 }
 
 SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(
     const std::string name, const DataType type, const Weights& beta, const Weights& gamma, const Weights& bias)
     : mLayerName(name)
     , mGammaDev(nullptr)
     , mBetaDev(nullptr)
     , mLd(beta.count)
     , mType(type)
     , mBiasDev(nullptr)
     , mParamsOnDevice(false)
 {
     assert(mLd > 0);
     assert(beta.count == gamma.count);
     assert(mType == nvinfer1::DataType::kFLOAT || mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kINT8);
     // mCfgType is the dataType for beta, gamma bias weights, always fp16 or fp32
     // mType is the plugin IO datatype, can be int8
     mCfgType = mType == DataType::kINT8 ? DataType::kHALF :  mType;
     mParamWordsize = getElementSize(mCfgType);
 
     mBeta.convertAndCopy(beta, mCfgType);
     mGamma.convertAndCopy(gamma, mCfgType);
 
     mHasBias = (bias.values != nullptr);
     if (mHasBias)
     {
         mBias.convertAndCopy(bias, mCfgType);
     }
 
 }
 
 SkipLayerNormVarSeqlenPlugin::SkipLayerNormVarSeqlenPlugin(const std::string name, const void* data, size_t length)
     : mLayerName(name)
     , mGammaDev(nullptr)
     , mBetaDev(nullptr)
     , mBiasDev(nullptr)
     , mParamsOnDevice(false)
 {
     BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin deserialize");
 
     // Deserialize in the same order as serialization
     deserialize_value(&data, &length, &mType);
     deserialize_value(&data, &length, &mCfgType);
     deserialize_value(&data, &length, &mLd);
     deserialize_value(&data, &length, &mHasBias);
 
     assert(mCfgType == nvinfer1::DataType::kFLOAT || mCfgType == nvinfer1::DataType::kHALF);
     mParamWordsize = getElementSize(mCfgType);
 
     const char* d = static_cast<const char*>(data);
     mBeta.convertAndCopy(d, mLd, mCfgType);
     mGamma.convertAndCopy(d, mLd, mCfgType);
     if (mHasBias)
     {
         mBias.convertAndCopy(d, mLd, mCfgType);
     }
 }
 
 // IPluginV2DynamicExt Methods
 IPluginV2DynamicExt* SkipLayerNormVarSeqlenPlugin::clone() const noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin clone");
 
     auto* p = new SkipLayerNormVarSeqlenPlugin(mLayerName, mType, mBeta, mGamma, mBias);
     p->initialize();
     p->setPluginNamespace(mNamespace.c_str());
     return p;
 }
 
 DimsExprs SkipLayerNormVarSeqlenPlugin::getOutputDimensions(
     int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
 {
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     assert(outputIndex == 0);
     // assert(inputs[0].nbDims == inputs[1].nbDims);
     return inputs[0];
 }
 
 bool SkipLayerNormVarSeqlenPlugin::supportsFormatCombination(
     int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
 {
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     assert(nbOutputs == 1);
 
     const PluginTensorDesc& in = inOut[pos];
 
     if(mType != in.type) return false;
     if (pos == 0)
     {
         // Since H = W = 1, we can report CHWx for any x
         if (mType == DataType::kINT8)
         {
             // won't work for hiddensize too small!
             TensorFormat myFmt = TensorFormat::kCHW32;
             if (mLd < 32)
             {
                 myFmt = TensorFormat::kCHW4;
                 BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW4 for LD=", mLd);
             }
             else
             {
                 BERT_DEBUG_VALUE("SkipLayerNormDQQ: TensorFormat CHW32 for LD=", mLd);
             }
             // TODO do we need to check if the vectorization divides mLd?
             return in.format == myFmt;
         }
         return in.format == TensorFormat::kLINEAR;
     }
     const PluginTensorDesc& prev = inOut[pos - 1];
 
     return in.format == prev.format;
 }
 
 void SkipLayerNormVarSeqlenPlugin::configurePlugin(
     const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept
 {
     // Validate input arguments
     assert(nbOutputs == 1);
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     if (mType == DataType::kFLOAT || mType == DataType::kHALF)
     {
         assert(mType == inputs[0].desc.type);
         // assert(mType == inputs[1].desc.type);
     }
     else
     {
         assert(mType == inputs[0].desc.type || DataType::kFLOAT == inputs[0].desc.type);
         // assert(mType == inputs[1].desc.type || DataType::kFLOAT == inputs[1].desc.type);
     }
     const auto& inDims0 = inputs[0].desc.dims;
     // const auto& inDims1 = inputs[1].desc.dims;
     // TRT_UNUSED inDims1;
     // assert(inDims0.nbDims == inDims1.nbDims);
 
     // assert(std::equal(inDims0.d, inDims0.d + inDims0.nbDims, inDims1.d));
 
     mCfgType = inputs[0].desc.type == DataType::kINT8 ? DataType::kHALF : inputs[0].desc.type;
 
     const auto paramType = getParamWordType(mCfgType);
     mParamWordsize = getElementSize(paramType);
 
     if (!mParamsOnDevice)
     {
         copyToDevice(mGamma, getWeightsSize(mGamma, paramType), mGammaDev);
         copyToDevice(mBeta, getWeightsSize(mBeta, paramType), mBetaDev);
         if (mHasBias)
         {
             copyToDevice(mBias, getWeightsSize(mBias, paramType), mBiasDev);
         }
         mParamsOnDevice = true;
     }
 }
 
 size_t SkipLayerNormVarSeqlenPlugin::getWorkspaceSize(
     const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
 {
     return 0;
 }
 
 int SkipLayerNormVarSeqlenPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
     const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
 {
     const int inputVolume = volume(inputDesc[0].dims);
     assert(inputVolume % mLd == 0 && "inconsistent dimensions");
     int status = -1;
     DataType iType = inputDesc->type;
 
     // Our plugin outputs only one tensor
     // Launch CUDA kernel wrapper and save its return value
     if (iType == DataType::kFLOAT)
     {
         const auto* const input = static_cast<const float*>(inputs[0]);
         // const auto* const skip = static_cast<const float*>(inputs[1]);
         auto* output = static_cast<float*>(outputs[0]);
         const auto* const bias = static_cast<const float*>(mBiasDev.get());
         const auto* const beta = static_cast<const float*>(mBetaDev.get());
         const auto* const gamma = static_cast<const float*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNorm<float, true>(
                 stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias);
         }
         else
         {
             status
                 = computeSkipLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias);
         }
     }
     else if (iType == DataType::kHALF)
     {
         const auto* const input = static_cast<const half*>(inputs[0]);
         // const auto* const skip = static_cast<const half*>(inputs[1]);
         auto* output = static_cast<half*>(outputs[0]);
         const auto* const bias = static_cast<const half*>(mBiasDev.get());
         const auto* const beta = static_cast<const half*>(mBetaDev.get());
         const auto* const gamma = static_cast<const half*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNorm<half, true>(
                 stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias);
         }
         else
         {
             status
                 = computeSkipLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias);
         }
     }
     else if (iType == DataType::kINT8)
     {
         const float dqScaleIn = inputDesc[0].scale;
         // const float dqScaleSkip = inputDesc[1].scale;
         const float qScale = 1.F / outputDesc[0].scale;
         const auto* const input = static_cast<const int8_t*>(inputs[0]);
         // const auto* const skip = static_cast<const int8_t*>(inputs[1]);
         auto* output = static_cast<int8_t*>(outputs[0]);
         const auto* const bias = static_cast<const half*>(mBiasDev.get());
         const auto* const beta = static_cast<const half*>(mBetaDev.get());
         const auto* const gamma = static_cast<const half*>(mGammaDev.get());
         if (mHasBias)
         {
             status = computeSkipLayerNormDQQ<true>(stream, static_cast<int>(mLd), inputVolume, input, beta, gamma,
                 output, bias, dqScaleIn, qScale);
         }
         else
         {
             status = computeSkipLayerNormDQQ<false>(
                 stream, static_cast<int>(mLd), inputVolume, input, beta, gamma, output, bias, dqScaleIn, qScale);
         }
     }
     else
     {
        std::cerr << "Unsupported type error, expected [kINT8,kHALF,kFLOAT], but received " << static_cast<int>(iType) << "." << std::endl;
         assert(false);
     }
     return status;
 }
 
 // IPluginV2Ext Methods
 DataType SkipLayerNormVarSeqlenPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept
 {
     assert(index == 0);
     // assert(nbInputs == 2);
     assert(nbInputs == 1);
     return inputTypes[0];
 }
 
 // IPluginV2 Methods
 const char* SkipLayerNormVarSeqlenPlugin::getPluginType() const noexcept
 {
     return SKIP_LAYER_NORM_NAME;
 }
 
 const char* SkipLayerNormVarSeqlenPlugin::getPluginVersion() const noexcept
 {
     return SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
 }
 
 int SkipLayerNormVarSeqlenPlugin::getNbOutputs() const noexcept
 {
     return 1;
 }
 int SkipLayerNormVarSeqlenPlugin::initialize() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin initialize");
     return 0;
 }
 
 void SkipLayerNormVarSeqlenPlugin::terminate() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin terminate");
 }
 
 size_t SkipLayerNormVarSeqlenPlugin::getSerializationSize() const noexcept
 {
     const size_t biasSize = mHasBias ? (mLd * mParamWordsize) : 0;
     return 2 * mParamWordsize * mLd + 2 * sizeof(DataType) + sizeof(mLd) + biasSize + sizeof(mHasBias);
 }
 
 void SkipLayerNormVarSeqlenPlugin::serialize(void* buffer) const noexcept
 {
     serialize_value(&buffer, mType);
     serialize_value(&buffer, mCfgType);
     serialize_value(&buffer, mLd);
     serialize_value(&buffer, mHasBias);
 
     char* d = static_cast<char*>(buffer);
     serFromDev(d, static_cast<char*>(mBetaDev.get()), mLd * mParamWordsize);
     serFromDev(d, static_cast<char*>(mGammaDev.get()), mLd * mParamWordsize);
     if (mHasBias)
     {
         serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * mParamWordsize);
     }
 }
 
 void SkipLayerNormVarSeqlenPlugin::destroy() noexcept
 {
     BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPlugin destroy");
     // This gets called when the network containing plugin is destroyed
     mGammaDev.reset(nullptr);
     mBetaDev.reset(nullptr);
     mBiasDev.reset(nullptr);
     delete this;
 }
 
 void SkipLayerNormVarSeqlenPlugin::setPluginNamespace(const char* libNamespace) noexcept
 {
     mNamespace = libNamespace;
 }
 
 const char* SkipLayerNormVarSeqlenPlugin::getPluginNamespace() const noexcept
 {
     return mNamespace.c_str();
 }
 
 /////////////////////////////////////////////////////////
 
 SkipLayerNormVarSeqlenPluginCreator::SkipLayerNormVarSeqlenPluginCreator()
 {
     mPluginAttributes.clear();
     mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
     mPluginAttributes.emplace_back(PluginField("beta", nullptr, PluginFieldType::kFLOAT32 , 256));
     mPluginAttributes.emplace_back(PluginField("gamma", nullptr, PluginFieldType::kFLOAT32 , 256));
     mPluginAttributes.emplace_back(PluginField("mLd", nullptr, PluginFieldType::kINT32, 1));
     mFC.nbFields = mPluginAttributes.size();
     mFC.fields = mPluginAttributes.data();
 }
 
 const char* SkipLayerNormVarSeqlenPluginCreator::getPluginName() const noexcept
 {
     return SKIP_LAYER_NORM_NAME;
 }
 
 const char* SkipLayerNormVarSeqlenPluginCreator::getPluginVersion() const noexcept
 {
     return SKIP_LAYER_NORM_VAR_SEQLEN_VERSION;
 }
 
 const PluginFieldCollection* SkipLayerNormVarSeqlenPluginCreator::getFieldNames() noexcept
 {
     return &mFC;
 }
 
 IPluginV2* SkipLayerNormVarSeqlenPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
 {
     try
     {
         BERT_DEBUG_MSG("SkipLayerNormVarSeqlenPluginCreator createPlugin");
 
         Weights beta{DataType::kFLOAT, nullptr, 0};
         Weights gamma{DataType::kFLOAT, nullptr, 0};
         Weights bias{DataType::kFLOAT, nullptr, 0};
         int typeId = -1;
 
         for (int i = 0; i < fc->nbFields; i++)
         {
             std::string field_name(fc->fields[i].name);
 
             if (field_name.compare("type_id") == 0)
             {
                 typeId = *static_cast<const int*>(fc->fields[i].data);
                 BERT_DEBUG_VALUE("Building typeId: ", typeId);
             }
 
             if (field_name.compare("beta") == 0)
             {
                 BERT_DEBUG_MSG("Building beta...");
                 beta.values = fc->fields[i].data;
                 beta.count = fc->fields[i].length;
                 beta.type = fieldTypeToDataType(fc->fields[i].type);
             }
 
             if (field_name.compare("gamma") == 0)
             {
                 BERT_DEBUG_MSG("Building gamma...");
                 gamma.values = fc->fields[i].data;
                 gamma.count = fc->fields[i].length;
                 gamma.type = fieldTypeToDataType(fc->fields[i].type);
             }
 
             if (field_name.compare("bias") == 0)
             {
                 BERT_DEBUG_MSG("Building bias...");
                 bias.values = fc->fields[i].data;
                 bias.count = fc->fields[i].length;
                 bias.type = fieldTypeToDataType(fc->fields[i].type);
             }
         }
         BERT_DEBUG_VALUE("Type ", typeId);
         BERT_DEBUG_VALUE("fc->nbFields ",fc->nbFields);
 
         if (typeId < 0 || typeId > 3)
         {
            std::cerr << "SkipLayerNorm: Invalid type ID: " << typeId << std::endl;
             std::cerr << "fc->nbFields : " << fc->nbFields<< std::endl;
         }
 
         if (beta.count <= 0 || beta.values == nullptr)
         {
            std::cerr << "SkipLayerNorm: invalid beta" << std::endl;
         }
 
         if (gamma.count <= 0 || gamma.values == nullptr)
         {
            std::cerr << "SkipLayerNorm: invalid gamma" << std::endl;
         }
 
         return new SkipLayerNormVarSeqlenPlugin(name, static_cast<DataType>(typeId), beta, gamma, bias);
     }
     catch (const std::exception& e)
     {
         caughtError(e);
     }
     return nullptr;
 }
 
 IPluginV2* SkipLayerNormVarSeqlenPluginCreator::deserializePlugin(
     const char* name, const void* serialData, size_t serialLength) noexcept
 {
     // This object will be deleted when the network is destroyed, which will
     // call SkipLayerNormVarSeqlenPlugin::destroy()
     try
     {
         return new SkipLayerNormVarSeqlenPlugin(name, serialData, serialLength);
     }
     catch (const std::exception& e)
     {
         caughtError(e);
     }
     return nullptr;
 }
 
 void SkipLayerNormVarSeqlenPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
 {
     mNamespace = libNamespace;
 }
 
 const char* SkipLayerNormVarSeqlenPluginCreator::getPluginNamespace() const noexcept
 {
     return mNamespace.c_str();
 }
 } // namespace bert

 

namespace bert
{

template <int TPB, int VPT, bool hasBias>
__global__ void skiplnDQQ(const int ld, const int8_t* input,  int8_t* output, const __half* beta,
    const __half* gamma, const __half* bias, const float dqScaleIn,  const float qScale)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    int8_t in_local[VPT];
    // int8_t skip_local[VPT];

    __half in_local_dq[VPT]; // dequantized input + skip + bias
    __half bias_local[VPT];  // bias and beta
    __half gamma_local[VPT];
    copy<sizeof(int8_t) * VPT>(&input[idx], in_local);
    // copy<sizeof(int8_t) * VPT>(&skip[idx], skip_local);
    copy<sizeof(__half) * VPT>(&bias[threadIdx.x * VPT], bias_local);
    __half2 loc = __floats2half2_rn(0.f, 0.f); // accumulator

    const __half rld = __half(1) / __half(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        // DQ input and skip
        const float tmp_in = in_local[it];
        // const float tmp_skip = skip_local[it];
        // in_local_dq[it] = dqScaleIn * tmp_in + dqScaleSkip * tmp_skip;
        in_local_dq[it] = dqScaleIn * tmp_in ;

        if (hasBias)
            in_local_dq[it] += bias_local[it];
        const __half tmp = rld * in_local_dq[it];
        const __half2 tmp2 = __halves2half2(tmp, tmp * in_local_dq[it]);
        loc = loc + tmp2;
    }
    // load parameters
    copy<sizeof(__half) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(__half) * VPT>(&gamma[threadIdx.x * VPT], gamma_local);

    using BlockReduce = cub::BlockReduce<__half2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ __half mu;     // mean
    __shared__ __half rsigma; // 1 / std.dev.

    const __half2 sum2 = BlockReduce(temp_storage).Reduce(loc, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = __low2half(sum2);
        rsigma = rsqrt(__high2half(sum2) - mu * mu );
    }
    __syncthreads();

    static_assert(VPT % 4 == 0, "");
    uint32_t out_local[VPT/4];
#pragma unroll
    for (int it = 0; it < VPT / 4; it++)
    {
        const float tmp0 = gamma_local[it*4+0] * (in_local_dq[it*4+0] - mu) * rsigma + bias_local[it*4+0];
        const float tmp1 = gamma_local[it*4+1] * (in_local_dq[it*4+1] - mu) * rsigma + bias_local[it*4+1];
        const float tmp2 = gamma_local[it*4+2] * (in_local_dq[it*4+2] - mu) * rsigma + bias_local[it*4+2];
        const float tmp3 = gamma_local[it*4+3] * (in_local_dq[it*4+3] - mu) * rsigma + bias_local[it*4+3];
        out_local[it] = float4_to_char4(tmp0 * qScale, tmp1 * qScale, tmp2 * qScale, tmp3 * qScale);
    }

    copy<sizeof(int8_t) * VPT>(out_local, &output[idx]);
}

template <typename T, int TPB, int VPT, bool hasBias>
__global__ void skipln_vec(
    const int ld, const T* input, T* output, const T* beta, const T* gamma, const T* bias)
{
    const int idx = ld * blockIdx.x + threadIdx.x * VPT;
    // 4 * 1024 * 4 * 2 Bytes = 16KB per block
    T in_local[VPT];
    //gyy: for gamma
    T skip_local[VPT];
    T bias_local[VPT];
    // T gamma_local[VPT];
    copy<sizeof(T) * VPT>(&input[idx], in_local);
    // copy<sizeof(T) * VPT>(&skip[idx], skip_local);
    copy<sizeof(T) * VPT>(&bias[threadIdx.x * VPT], bias_local);
    T local = 0.f;
    T local2 = 0.f;

    const T rld = T(1) / T(ld);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        // in_local[it] += skip_local[it];
        if (hasBias)
            in_local[it] += bias_local[it];
        const T tmp = rld * in_local[it];
        local += tmp;
        local2 += tmp * in_local[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], bias_local);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], skip_local);

    using BlockReduce = cub::BlockReduce<kvp<T>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T mu;     // mean
    __shared__ T rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(kvp<T>(local, local2), cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu );
    }
    __syncthreads();
    ///*
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        in_local[it] = skip_local[it] * (in_local[it] - mu) * rsigma + bias_local[it];
    }
    /* */

    copy<sizeof(T) * VPT>(in_local, &output[idx]);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernelSmall(
    const int ld, const T* input, const T* beta, const T* gamma, T* output, const T* bias)
{

    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);
    const int idx = offset + threadIdx.x;
    T val = 0;

    if (threadIdx.x < ld)
    {

        // val = input[idx] + skip[idx];
        val = input[idx];
        if (hasBias)
        {
            val += bias[threadIdx.x];
        }

        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
    }

    layerNormSmall<T, T, TPB>(val, threadData, ld, idx, beta, gamma, output);
}

template <typename T, unsigned TPB, bool hasBias>
__global__ void skipLayerNormKernel(
    const int ld, const T* input,  const T* beta, const T* gamma, T* output, const T* bias)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        // T val = T(input[idx]) + T(skip[idx]);
        T val = T(input[idx]);

        if (hasBias)
        {
            val += T(bias[i]);
        }
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, beta, gamma, output);
}

template <bool hasBias>
int computeSkipLayerNormDQQ(cudaStream_t stream, const int ld, const int n, const int8_t* input, 
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
     const float qScale)
{
    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);

    const int gridSize = n / ld;
    // we're limited by the size of the parameters, i.e. 8-wide instead of 16
    constexpr int VPT = 16 / sizeof(__half);
    if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias, dqScaleIn, qScale);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        skiplnDQQ<TPB, VPT, hasBias>
            <<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias, dqScaleIn, qScale);
    }
    else
    {
        // TODO need to implement this
        std::cerr << "SkipLayerNormDQQ - FATAL: unsupported hidden layer size: " << ld << std::endl;
        exit(0);
    }
    CHECK(cudaPeekAtLastError());

    return 0;
}

template <typename T, bool hasBias>
int computeSkipLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, const T* beta,
    const T* gamma, T* output, const T* bias)
{

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;
    constexpr int VPT = 16 / sizeof(T);
    if (ld <= 32)
    {
        constexpr int blockSize = 32;
        skipLayerNormKernelSmall<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, beta, gamma, output, bias);
    }
    else if (ld == 768)
    {
        constexpr int TPB = 768 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias);
    }
    else if (ld == 1024)
    {
        constexpr int TPB = 1024 / VPT;
        skipln_vec<T, TPB, VPT, hasBias><<<gridSize, TPB, 0, stream>>>(ld, input, output, beta, gamma, bias);
    }
    else
    {
        constexpr int blockSize = 256;
        skipLayerNormKernel<T, blockSize, hasBias>
            <<<gridSize, blockSize, 0, stream>>>(ld, input, beta, gamma, output, bias);
    }
    CHECK(cudaPeekAtLastError());

    return 0;
}

template int computeSkipLayerNormDQQ<true>(cudaStream_t stream, const int ld, const int n, const int8_t* input, 
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
     const float qScale);
template int computeSkipLayerNormDQQ<false>(cudaStream_t stream, const int ld, const int n, const int8_t* input, 
    const __half* beta, const __half* gamma, int8_t* output, const __half* bias, const float dqScaleIn,
     const float qScale);

template int computeSkipLayerNorm<float, true>(cudaStream_t, const int, const int, const float*, const float*,  const float*, float*, const float*);
template int computeSkipLayerNorm<float, false>(cudaStream_t, const int, const int, const float*, const float*,  const float*, float*, const float*);
template int computeSkipLayerNorm<half, true>(cudaStream_t, const int, const int, const half*, const half*,  const half*, half*, const half*);
template int computeSkipLayerNorm<half, false>(cudaStream_t, const int, const int, const half*, const half*,  const half*, half*, const half*);

} // namespace bert

#endif // CUDA_VERSION >= 10010
