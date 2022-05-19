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

#include "MHAPlugin.h"
// #include "decoder_masked_multihead_attention.h"
#include "NvInfer.h"
// #include "decoder_masked_multihead_attention_utils.h"
// #include "bfloat16_fallback_kenrels.cuh"

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

// MHA plugin specific constants
namespace
{
const char* MHA_PLUGIN_VERSION{"1"};
const char* MHA_PLUGIN_NAME{"MultiHeadAttn"};
} // namespace

// Static class fields initialization
PluginFieldCollection MHAPluginCreator::mFC{};
std::vector<PluginField> MHAPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MHAPluginCreator);


// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

MHAPlugin::MHAPlugin(const std::string name, bool isCrossAtten)
    : mLayerName(name),
    isCrossAtten(isCrossAtten)
{
    // printf("%s \n", __FUNCTION__);
}

MHAPlugin::MHAPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // printf("%s \n", __FUNCTION__);
    // Deserialize in the same order as serialization
    const char* d = static_cast<const char*>(data);
    const char* a = d;

    isCrossAtten = readFromBuffer<bool>(d);
    // mMHAMax = readFromBuffer<float>(d);
    // printf("MHAPlugin::MHAPlugin, readFromBuffer isCrossAtten: %s \n", isCrossAtten?"true":"false");

    assert(d == (a + length));
}

const char* MHAPlugin::getPluginType() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return MHA_PLUGIN_NAME;
}

const char* MHAPlugin::getPluginVersion() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return MHA_PLUGIN_VERSION;
}

int MHAPlugin::getNbOutputs() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return 1;
}

DimsExprs MHAPlugin::getOutputDimensions(int32_t        outputIndex,
                                    const DimsExprs *   inputs,
                                    int32_t             nbInputs,
                                    IExprBuilder &      exprBuilder) noexcept
{
    // printf("%s \n", __FUNCTION__);
    // Validate input arguments
    assert(nbInputs == 13);
    assert(outputIndex == 0);

    // MHAping doesn't change input dimension, so output Dims will be the same as input Dims
    return inputs[0];
}

int MHAPlugin::initialize() noexcept
{

    return 0;
}

void MHAPlugin::terminate() noexcept
{
    // if(isCrossAtten)
    //     delete cross_attn;
    // printf("%s \n", __FUNCTION__);
    // delete cublas_wrapper;
    // printf("%s \n", __FUNCTION__);
    // // allocator = nullptr;
    // delete cublasAlgoMap_;
    // delete cublasWrapperMutex_;
    // printf("%s \n", __FUNCTION__);
    // delete allocator;
    // printf("%s \n", __FUNCTION__);
    // printf("%s \n", __FUNCTION__);

}

size_t MHAPlugin::getWorkspaceSize(const PluginTensorDesc *    inputDesc,
                            int32_t                     nbInputs,
                            const PluginTensorDesc *    outputs,
                            int32_t                     nbOutputs 
                            )   const noexcept
{
    // printf("%s \n", __FUNCTION__);
    int batch_size  = inputDesc[0].dims.d[0];    // B
    int seq_len0    = inputDesc[0].dims.d[1];    // T0 for q
    int d_model     = inputDesc[0].dims.d[2];    // D
    int seq_len1    = isCrossAtten ? inputDesc[1].dims.d[1] : seq_len0;    // T1 for k v 
    
    int dataNum = d_model*(seq_len0*4 + seq_len1*4 +seq_len0*seq_len1*8);    //  q_buf, out_buf, k_buf, v_buf
    size_t workspaceSize = batch_size*dataNum*sizeof(float);
    return workspaceSize;
}
void MHAPlugin::attachToContext(cudnnContext * cudnn_handle,
                        cublasContext * cublas_handle,
                        IGpuAllocator * gpu_allocator
                        )noexcept
{
    // printf("%s \n", __FUNCTION__);
    cublasHandle_ = cublas_handle;
    gpu_allocator_ = gpu_allocator;
}

int MHAPlugin::pre_enqueue(cudaStream_t stream) noexcept
{
    // printf("%s \n", __FUNCTION__);
    // cublasSetStream(cublasltHandle_, stream);

    return 0;
}

template<typename T>
void dump2Txt(T* src, int N, std::string fname)
{
    return ;
    FILE * pFile;
    pFile = fopen(fname.c_str(),"w");
    
    std::cout << fname ; 
    T *dst_cpu = new T[N];
    cudaMemcpy(dst_cpu, src, N*sizeof(T), cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++)
    {
        fprintf(pFile, "%f\n", (float)(dst_cpu[i]));
    }
    std::cout<< "       done" << std::endl;
    fclose(pFile);
    delete []dst_cpu;
}

int MHAPlugin::enqueue(const PluginTensorDesc*  inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void *const *     inputs,
                    void *const *           outputs,
                    void *                  workspace,
                    cudaStream_t            stream) noexcept
{
    cudaStreamSynchronize(stream);
    //  input : q, enc_in, enc_lens, qw, qb, kw, kb, vw, vb, lw, lb
    int status = 0;
    const int batch_size  = inputDesc[0].dims.d[0];    // B
    const int seq_len0    = inputDesc[0].dims.d[1];    // T0 for q
    const int d_model     = inputDesc[0].dims.d[2];    // D
    const int seq_len1    = isCrossAtten ? inputDesc[1].dims.d[1] : seq_len0;    // T1 for k v
    const int head_num    = 4 ;
    const int size_per_head = 64 ;
    int widx = 0;
    float *query_in              = (float*)(inputs[widx++]);
    float *enc_in                = (float*)(inputs[widx++]);
    int   *enc_mask              = (int*)(inputs[widx++]);
    float *query_weight_kernel   = (float*)(inputs[widx++]);
    float *query_weight_bias     = (float*)(inputs[widx++]);
    float *key_weight_kernel     = (float*)(inputs[widx++]);
    float *key_weight_bias       = (float*)(inputs[widx++]);
    float *value_weight_kernel   = (float*)(inputs[widx++]);
    float *value_weight_bias     = (float*)(inputs[widx++]);
    float *output_weight_kernel  = (float*)(inputs[widx++]);
    float *output_weight_bias    = (float*)(inputs[widx++]);
    float *layer_norm_gamma      = (float*)(inputs[widx++]);
    float *layer_norm_beta       = (float*)(inputs[widx++]);
    int ws_offset  = 0;
    float* q_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* q_share = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* qk_buf  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len0*seq_len1;
    float* out_buf = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* k_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* k_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* v_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* v_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;

    // cudaStreamSynchronize(stream);
    cublasSetStream(cublasHandle_, stream);

    invokeGeneralLayerNorm(q_share,     //  float* out,
                        query_in,       //  const float* input,
                        layer_norm_gamma,   //  const float* gamma,
                        layer_norm_beta,    //  const float* beta,
                        batch_size*seq_len0,    //  const int m,
                        d_model,                //  const int n,
                        stream,             //  cudaStream_t stream,
                        0                   //  int opt_version
                    );
    int enc_mask_num = batch_size/10;
    if (! isCrossAtten){     //  self attn
        enc_in = q_share;
        enc_mask_num = batch_size;
    }
    // printf("debug  inputs %d %d %d, %d %d %d \n", batch_size, seq_len0, d_model, 
    //                 inputDesc[1].dims.d[0], inputDesc[1].dims.d[1], inputDesc[1].dims.d[2]);

    // dump2Txt((float*)(query_in), batch_size*seq_len0*d_model, "dump_trt_input/"+mLayerName+"_query_in.txt");
    // dump2Txt((float*)(q_share), batch_size*seq_len0*d_model, "dump_trt_input/"+mLayerName+"_query_layer_norm.txt");
    // dump2Txt((float*)(enc_in), batch_size*seq_len1*d_model, "dump_trt_input/"+mLayerName+"_enc_in.txt");
    // dump2Txt((int*)(enc_mask), enc_mask_num, "dump_trt_input/"+mLayerName+"_enc_mask.txt");
    // dump2Txt((float*)(query_weight_kernel), d_model*d_model, "dump_trt_input/"+mLayerName+"_query_kernel.txt");
    // dump2Txt((float*)(query_weight_bias)  , d_model,         "dump_trt_input/"+mLayerName+"_query_bias.txt");
    // dump2Txt((float*)(key_weight_kernel), d_model*d_model, "dump_trt_input/"+mLayerName+"_key_kernel.txt");
    // dump2Txt((float*)(key_weight_bias)  , d_model,         "dump_trt_input/"+mLayerName+"_key_bias.txt");
    // dump2Txt((float*)(value_weight_kernel), d_model*d_model, "dump_trt_input/"+mLayerName+"_value_kernel.txt");
    // dump2Txt((float*)(value_weight_bias)  , d_model,         "dump_trt_input/"+mLayerName+"_value_bias.txt");
    // dump2Txt((float*)(output_weight_kernel), d_model*d_model, "dump_trt_input/"+mLayerName+"_linear_kernel.txt");
    // dump2Txt((float*)(output_weight_bias)  , d_model,         "dump_trt_input/"+mLayerName+"_linear_bias.txt");
    // dump2Txt((float*)(layer_norm_gamma)  , d_model,         "dump_trt_input/"+mLayerName+"_layer_norm_gamma.txt");
    // dump2Txt((float*)(layer_norm_beta)  , d_model,         "dump_trt_input/"+mLayerName+"_layer_norm_beta.txt");

    const float alpha = 1.0, beta = 0.0;
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len0,    //  n
        d_model,                //  k
        &alpha, query_weight_kernel, d_model,
        (const float*)(q_share), d_model,
        &beta, q_buf, d_model
    );
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len1,    //  n
        d_model,                //  k
        &alpha, key_weight_kernel, d_model,
        (const float*)(enc_in), d_model,
        &beta, k_buf, d_model
    );
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len1,    //  n
        d_model,                //  k
        &alpha, value_weight_kernel, d_model,
        (const float*)(enc_in), d_model,
        &beta, v_buf, d_model
    );
    // cudaStreamSynchronize(stream);
    // dump2Txt((float*)(q_share), batch_size*seq_len0*d_model, "dump_trt/"+mLayerName+"_q_in.txt");
    // dump2Txt((float*)(enc_in), batch_size*seq_len1*d_model, "dump_trt/"+mLayerName+"_enc_in.txt");
    // dump2Txt(q_buf, batch_size*seq_len0*d_model, "dump_trt/"+mLayerName+"_q_buf.txt");
    // dump2Txt(k_buf, batch_size*seq_len1*d_model, "dump_trt/"+mLayerName+"_k_buf.txt");
    // dump2Txt(v_buf, batch_size*seq_len1*d_model, "dump_trt/"+mLayerName+"_v_buf.txt");

    const int qm = batch_size * seq_len0;
    const int n = head_num * size_per_head;
    dim3 block(384);
    dim3 gridq((int)(ceil(1.0 * qm * n / 384)));
    add_fusedQKV_bias_transpose_kernel<<<gridq, block, 0, stream>>>(
        q_share, q_buf, query_weight_bias, batch_size, seq_len0, head_num, size_per_head);
    const int kvm = batch_size * seq_len1;
    dim3 gridkv((int)(ceil(1.0 * kvm * n / 384)));
    add_fusedQKV_bias_transpose_kernel<<<gridq, block, 0, stream>>>(
        k_bias, k_buf, key_weight_bias, batch_size, seq_len1, head_num, size_per_head);
    add_fusedQKV_bias_transpose_kernel<<<gridq, block, 0, stream>>>(
        v_bias, v_buf, value_weight_bias, batch_size, seq_len1, head_num, size_per_head);
    // cudaStreamSynchronize(stream);
    // dump2Txt(q_share, batch_size*seq_len0*d_model, "dump_trt/"+mLayerName+"_q_bias_trans.txt");
    // dump2Txt(k_bias, batch_size*seq_len1*d_model, "dump_trt/"+mLayerName+"_k_bias_trans.txt");
    // dump2Txt(v_bias, batch_size*seq_len1*d_model, "dump_trt/"+mLayerName+"_v_bias_trans.txt");

    cublasSgemmStridedBatched(
            cublasHandle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len1,               //m,
            seq_len0,               //n,
            size_per_head,          //k,
            &alpha,
            k_bias,                     //A,
            size_per_head,              //lda,
            seq_len1*size_per_head,     //strideA,
            q_share,                     //B,
            size_per_head,              //ldb,
            seq_len0*size_per_head,     //strideB,
            &beta,
            qk_buf,                 //C,
            seq_len1,               //ldc,
            seq_len0*seq_len1,      //strideC,
            batch_size*head_num     //batchCount
        );
    // cudaStreamSynchronize(stream);
    // dump2Txt(qk_buf, batch_size*head_num*seq_len0*seq_len1, "dump_trt/"+mLayerName+"_qk_buf.txt");
    
    float scalar = 1 / sqrtf(size_per_head * 1.0f);
    invokeMaskedSoftMax(qk_buf,
                        qk_buf,
                        enc_mask,
                        isCrossAtten,
                        batch_size,
                        seq_len0,
                        seq_len1,
                        head_num,
                        scalar,
                        stream
        );
    // cudaStreamSynchronize(stream);
    // dump2Txt(qk_buf, batch_size*head_num*seq_len0*seq_len1, "dump_trt/"+mLayerName+"_softmax.txt");

    cublasSgemmStridedBatched(
            cublasHandle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            size_per_head,          //m,
            seq_len0,               //n,
            seq_len1,               //k,
            &alpha,
            v_bias,                     //A,
            size_per_head,              //lda,
            seq_len1*size_per_head,     //strideA,
            qk_buf,                     //B,
            seq_len1,                   //ldb,
            seq_len0*seq_len1,          //strideB,
            &beta,
            q_buf,                  //C,
            size_per_head,          //ldc,
            seq_len0*size_per_head, //strideC,
            batch_size*head_num     //batchCount
        );
    // cudaStreamSynchronize(stream);
    // dump2Txt(q_buf, batch_size*head_num*seq_len0*size_per_head, "dump_trt/"+mLayerName+"_qkv.txt");

    invokeTransposeQKV(
        out_buf, q_buf, batch_size, seq_len0, head_num, size_per_head, stream);
    // cudaStreamSynchronize(stream);
    // dump2Txt(out_buf, batch_size*head_num*seq_len0*size_per_head, "dump_trt/"+mLayerName+"_qkv_trans.txt");
    // dump2Txt(output_weight_kernel, d_model*d_model, "dump_trt/"+mLayerName+"_lw.txt");

    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len0,    //  n
        d_model,                //  k
        &alpha, output_weight_kernel, d_model,
        (const float*)(out_buf), d_model,
        &beta, (float*)(outputs[0]), d_model
    );
    // cudaStreamSynchronize(stream);
    // dump2Txt((float*)(outputs[0]), batch_size*head_num*seq_len0*size_per_head, "dump_trt/"+mLayerName+"_out_linear.txt");
    // add_bias_kernel<<<gridq, block, 0, stream>>>(
    //     (float*)(outputs[0]), (float*)(outputs[0]), output_weight_bias, 
    //     batch_size, seq_len0, d_model);
    invokeAddBiasResidual((float*)(outputs[0]),     //  T* output, 
                        query_in,                   //  const T* input, 
                        output_weight_bias,         //  const T* bias, 
                        batch_size * seq_len0,       //  const int m, 
                        d_model,                    //  const int n, 
                        stream
                );
    // cudaStreamSynchronize(stream);
    // dump2Txt((float*)(outputs[0]), batch_size*head_num*seq_len0*size_per_head, "dump_trt/"+mLayerName+"_out_final.txt");

    return status;
}

size_t MHAPlugin::getSerializationSize() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    size_t ssize = 0;
    ssize += sizeof(bool);
    return ssize;
}

void MHAPlugin::serialize(void* buffer) const noexcept
{
    // printf("%s \n", __FUNCTION__);
    char* d = static_cast<char*>(buffer);
    const char* a = d;

    writeToBuffer(d, isCrossAtten);
    // // writeToBuffer(d, mMHAMax);

    assert(d == a + getSerializationSize());
}

bool MHAPlugin::supportsFormatCombination(int32_t               pos,
                                        const PluginTensorDesc *inOut,
                                        int32_t                 nbInputs,
                                        int32_t                 nbOutputs 
                                        ) noexcept
{
    // std::cout <<"pos " << pos << " format " << (int)inOut[pos].format 
    //     << " type " << (int)inOut[pos].type << std::endl;
    // printf("%s \n", __FUNCTION__);
    switch (pos)
    {
    case 2:
        return inOut[pos].type == DataType::kINT32;
        break;
    
    default:
        break;
    }
    return inOut[pos].type == DataType::kFLOAT;
}


void MHAPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    // printf("%s \n", __FUNCTION__);
    delete this;
}

IPluginV2DynamicExt* MHAPlugin::clone() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    auto plugin = new MHAPlugin(mLayerName, isCrossAtten);
    plugin->setPluginNamespace(mNamespace.c_str());
    // printf("clone cublasHandle_ %p \n ", plugin->cublasHandle_);
    return plugin;
}

void MHAPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    // printf("%s \n", __FUNCTION__);
    mNamespace = libNamespace;
}

const char* MHAPlugin::getPluginNamespace() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return mNamespace.c_str();
}

MHAPluginCreator::MHAPluginCreator()
{
    // Describe MHAPlugin's required PluginField arguments
    // mPluginAttributes.emplace_back(PluginField("MHAMin", nullptr, PluginFieldType::kFLOAT32, 1));
    // printf("%s \n", __FUNCTION__);
    mPluginAttributes.emplace_back(PluginField("AttentionType", nullptr, PluginFieldType::kCHAR, 4));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MHAPluginCreator::getPluginName() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return MHA_PLUGIN_NAME;
}

const char* MHAPluginCreator::getPluginVersion() const noexcept
{
    // printf("%s \n", __FUNCTION__);
    return MHA_PLUGIN_VERSION;
}

const PluginFieldCollection* MHAPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* MHAPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    // printf("createPlugin MHAPlugin, nbFields: %d \n", fc->nbFields);

    // Parse fields from PluginFieldCollection
    // assert(fc->nbFields == 2);
    bool isCrossAtten = false;
    for (int i = 0; i < fc->nbFields; i++)
    {
        // std::cout << fields[i].name <<std::endl;
        if (strcmp(fields[i].name, "AttentionType") == 0)
        {
            assert(fields[i].type == PluginFieldType::kCHAR);
            //  0: self attention, 1: cross attention
            if(strcmp(reinterpret_cast<const char*>(fields[i].data), "cross") == 0)
                isCrossAtten = true;
            // printf("createPlugin AttentionType : %s , %d\n", 
            //     reinterpret_cast<const char*>(fields[i].data), 
            //     isCrossAtten);
        }
    }
    return new MHAPlugin(name, isCrossAtten);
}

IPluginV2DynamicExt* MHAPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MHAPlugin::destroy()
    return new MHAPlugin(name, serialData, serialLength);
}

void MHAPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MHAPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
