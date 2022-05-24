/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUSTOM_LAYER_NORM_PLUGIN_H
#define CUSTOM_LAYER_NORM_PLUGIN_H

#include "NvInferPlugin.h"
// #include "decoder_masked_multihead_attention.h"
#include "cublas.h"
#include "cublas_v2.h"
#include <string>
#include <vector>

using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class LayerNormPlugin : public IPluginV2DynamicExt 
{
public:
    LayerNormPlugin(const std::string name);

    LayerNormPlugin(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make LayerNormPlugin without arguments, so we delete default constructor.
    LayerNormPlugin() = delete;

    int getNbOutputs() const noexcept override;

    DimsExprs getOutputDimensions(int32_t             outputIndex,
                            const DimsExprs *   inputs,
                            int32_t             nbInputs,
                            IExprBuilder &      exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const PluginTensorDesc *    inputs,
                            int32_t                     nbInputs,
                            const PluginTensorDesc *    outputs,
                            int32_t                     nbOutputs 
                            )   const noexcept override;

    void attachToContext(cudnnContext * cudnn_handle,
                        cublasContext * cublas_handle,
                        IGpuAllocator * gpu_allocator
                        )noexcept override;

    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void *const *     inputs,
                void *const *           outputs,
                void *                  workspace,
                cudaStream_t            stream) noexcept override;

    bool supportsFormatCombination(int32_t  pos,
                                const PluginTensorDesc *inOut,
                                int32_t                 nbInputs,
                                int32_t                 nbOutputs 
                                ) noexcept override;

    void configurePlugin(const DynamicPluginTensorDesc* in, 
                int32_t nbInputs,         
                const DynamicPluginTensorDesc* out, 
                int32_t nbOutputs) noexcept override
    {
        return ;
    }

    nvinfer1::DataType getOutputDataType(int32_t    index,
                                    nvinfer1::DataType const *inputTypes,
                                    int32_t nbInputs 
                                    ) const noexcept override
    {
        return inputTypes[0];
    }

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    const std::string mLayerName;
    size_t mInputVolume;
    std::string mNamespace;
    cublasHandle_t cublasHandle_;
    IGpuAllocator * gpu_allocator_;

};

class LayerNormPluginCreator : public IPluginCreator
{
public:
    LayerNormPluginCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif
