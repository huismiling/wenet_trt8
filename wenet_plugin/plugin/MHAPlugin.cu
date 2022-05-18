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


template<typename T>
__global__ void transpose_4d_batch_major_mem_q_cache(
    T* v_dst, const T* v_src, const int batch_size, const int seq_length, const int d_model)
{
    // B, L,Dm -> L, B, Dm
    const int batch_id = blockIdx.y;
    const int seq_id = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * d_model * seq_length
                                                  + seq_id * d_model);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + seq_id * batch_size * d_model
                                            + batch_id * d_model);

    // idx is over output dimension L * size_per_head / x for values
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    const int size_limit = d_model / X_ELEMS;
    if (out_idx >= size_limit) {
        return;
    }

    val_dst[out_idx] = val_src[out_idx];
}

template<typename T>
void transpose_3d_102_memory_kernelLauncher(T* dst,
                                            const T* src,
                                            const int local_batch_size,
                                            const int max_seq_len,
                                            const int d_model,
                                            cudaStream_t stream)
{
    constexpr int block_sz = 128;

    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    int size = d_model / x;
    dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, max_seq_len);

    transpose_4d_batch_major_mem_q_cache<<<grid, block_sz, 0, stream>>>(
            dst, src, local_batch_size, max_seq_len, d_model);
}

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* qkv_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head)
{
    // QKV: [m, n]
    // qkv_bias: [n]
    // q_buf, k_buf, v_buf: [batch, head_num, seq_len, size_per_head]

    const int n = head_num * size_per_head;
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * n;
         index += gridDim.x * blockDim.x) {
        int bias_id = index % (n);
        T val = ldg(&QKV[index]) + ldg(&qkv_bias[bias_id]);

        int tmp_index = index;
        const int target_batch_id = tmp_index / (seq_len * n);
        tmp_index -= target_batch_id * seq_len * n;     //  current batch index
        const int seq_id = tmp_index / (n);             //  current seq id
        tmp_index -= seq_id * n;                        //  current batch, seq, nindex
        const int head_id = tmp_index / size_per_head;
        const int size_id = tmp_index - head_id * size_per_head;

        qkv_buf[target_batch_id * head_num * seq_len * size_per_head 
                + head_id * seq_len * size_per_head
                + seq_id * size_per_head + size_id] = val;
    }
}
template<typename T>
__global__ void add_bias_kernel(T* qkv_buf,
                                const T* __restrict QKV,
                                const T* __restrict qkv_bias,
                                const int batch_size,
                                const int seq_len,
                                const int n)
{
    // QKV: [m, n]
    // qkv_bias: [n]

    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * n;
         index += gridDim.x * blockDim.x) {
        int bias_id = index % (n);
        T val = ldg(&QKV[index]) + ldg(&qkv_bias[bias_id]);
        qkv_buf[index] = val;
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_k_cache(
    T* k_dst, const T* k_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;

    auto key_src = reinterpret_cast<const uint4*>(k_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto key_dst = reinterpret_cast<uint4*>(k_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size_per_head_div_x = size_per_head / X_ELEMS;
    if (out_idx >= size_per_head_div_x * max_seq_len) {
        return;
    }

    int idx = out_idx;
    const int k_seq_len_id = idx % max_seq_len;
    idx = (idx - k_seq_len_id) / max_seq_len;
    const int k_head_size_id = idx % size_per_head_div_x;

    if (k_seq_len_id < seq_len) {
        key_dst[out_idx] = key_src[k_seq_len_id * size_per_head_div_x + k_head_size_id];
    }
}

template<typename T>
__global__ void transpose_4d_batch_major_v_cache(
    T* v_dst, const T* v_src, const int head_num, const int size_per_head, const int seq_len, const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    // 16 byte loads will handle "x" dimension
    auto val_src = reinterpret_cast<const uint4*>(v_src + batch_id * head_num * size_per_head * seq_len
                                                  + head_id * size_per_head * seq_len);
    auto val_dst = reinterpret_cast<uint4*>(v_dst + batch_id * head_num * size_per_head * max_seq_len
                                            + head_id * size_per_head * max_seq_len);

    // idx is over output dimension L * size_per_head / x for values
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int X_ELEMS = (sizeof(T) == 4) ? 4 : 8;
    const int size_per_head_div_x = size_per_head / X_ELEMS;

    if (idx >= size_per_head_div_x * seq_len) {
        return;
    }

    val_dst[idx] = val_src[idx];
}

template<typename T>
void invokeTranspose4dBatchMajor(T* k_dst,
                                 T* v_dst,
                                 const T* k_src,
                                 const T* v_src,
                                 const int local_batch_size,
                                 const int seq_len,
                                 const int max_seq_len,
                                 const int size_per_head,
                                 const int local_head_num,
                                 cudaStream_t stream)
{
    constexpr int block_sz = 128;
    constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    int size = max_seq_len * size_per_head / x;
    dim3 grid((size + block_sz - 1) / block_sz, local_batch_size, local_head_num);
    dim3 grid_v((seq_len * size_per_head / x + block_sz - 1) / block_sz, local_batch_size, local_head_num);

    transpose_4d_batch_major_k_cache<<<grid, block_sz, 0, stream>>>(
        k_dst, k_src, local_head_num, size_per_head, seq_len, max_seq_len);

    transpose_4d_batch_major_v_cache<<<grid_v, block_sz, 0, stream>>>(
        v_dst, v_src, local_head_num, size_per_head, seq_len, max_seq_len);
}

static const float HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}
// TODO(bhsueh) Rename the softmax_kernel_v4 to softmax_kernel
template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void softmax_kernel_v4(T* qk_buf_,
                                  const T_IN* qk_buf_src,
                                  const int* attr_mask,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  const int kv_len,
                                  const T scalar)
{
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        int mask_offset = blockIdx.y/10;
        int enc_len = static_cast<int>(ldg(&attr_mask[mask_offset]));
        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * kv_len + blockDim.x * i + threadIdx.x;

            float qk = static_cast<float>(qk_buf_src[qk_offset]);


            float mask_val = (blockDim.x * i + threadIdx.x)<enc_len? 0.f : -10000.0f;

            data[i] = qk * static_cast<float>(scalar) + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * kv_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}
/*
template<typename T, int ITEMS_PER_THREAD>
__global__ void softmax_kernel_v4_half2(
    T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        T2 data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            int mask_offset = (blockIdx.y * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i + threadIdx.x;

            T2 qk = qk_buf_half2[qk_offset];
            T2 mask_val = ldg(&attr_mask_half2[mask_offset]);
            mask_val = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val), float2type2<T2>(-10000.0f));

            data[i] = hadd2<T2>(hmul2<T2>(qk, type2type2<T, T2>(scalar)), mask_val);

            local_max = fmax(local_max, fmax((float)data[i].x, (float)data[i].y));
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            data[i] = hexp2<T2>(hsub2<T2>(data[i], float2type2<T2>(s_max)));
            local_sum += (float)(data[i].x + data[i].y);
        }

        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);

        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            qk_offset = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * (seq_len / 2) + blockDim.x * i
                        + threadIdx.x;
            qk_buf_half2[qk_offset] = hmul2<T2>(data[i], float2type2<T2>(s_mean));
        }
    }
}

template<typename T, int ITEMS_PER_THREAD, int NUM>
__global__ void softmax_kernel_v5_half2(
    T* qk_buf_, const T* attr_mask, const int batch_size, const int head_num, const int seq_len, const T scalar)
{
    using T2 = typename TypeConverter<T>::Type;
    T2* qk_buf_half2 = (T2*)qk_buf_;
    const T2* attr_mask_half2 = (const T2*)attr_mask;

    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x * NUM) {
        T2 data[NUM][ITEMS_PER_THREAD];

        int qk_offset[NUM];

        __shared__ float s_sum[NUM], s_max[NUM];
        float local_max[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_max[j] = -1e20f;
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
            int mask_offset[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
                mask_offset[j] =
                    (blockIdx.y * seq_len + seq_id + j * gridDim.x) * (seq_len / 2) + blockDim.x * i + threadIdx.x;
            }

            T2 mask_val[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = ldg(&attr_mask_half2[mask_offset[j]]);
            }

            T2 qk[NUM];
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk[j] = qk_buf_half2[qk_offset[j]];
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                mask_val[j] = hmul2<T2>(hsub2<T2>(float2type2<T2>(1.0f), mask_val[j]), float2type2<T2>(-10000.0f));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hadd2<T2>(hmul2<T2>(qk[j], type2type2<T, T2>(scalar)), mask_val[j]);
                local_max[j] = fmax(local_max[j], fmax((float)data[j][i].x, (float)data[j][i].y));
            }
        }

        if (blockDim.x <= 32) {
            warpReduceMaxV2<float, NUM>(local_max);
        }
        else {
            blockReduceMaxV2<float, NUM>(local_max);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_max[j] = local_max[j];
            }
        }
        __syncthreads();

        float local_sum[NUM];
#pragma unroll
        for (int j = 0; j < NUM; j++) {
            local_sum[j] = {0.f};
        }

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                data[j][i] = hexp2<T2>(hsub2<T2>(data[j][i], float2type2<T2>(s_max[j])));
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                local_sum[j] += (float)(data[j][i].x + data[j][i].y);
            }
        }

        if (blockDim.x <= 32) {
            warpReduceSumV2<float, NUM>(local_sum);
        }
        else {
            blockReduceSumV2<float, NUM>(local_sum);
        }

        if (threadIdx.x == 0) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                s_sum[j] = __fdividef(1.0f, local_sum[j] + 1e-6f);
            }
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < (seq_len / 2) && i < ITEMS_PER_THREAD; i++) {
#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_offset[j] = ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id + j * gridDim.x) * (seq_len / 2)
                               + blockDim.x * i + threadIdx.x;
            }

#pragma unroll
            for (int j = 0; j < NUM; j++) {
                qk_buf_half2[qk_offset[j]] = hmul2<T2>(data[j][i], float2type2<T2>(s_sum[j]));
            }
        }
    }
}

*/
#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel_v4<ITEMS_PER_THREAD, T, T_IN>                                                                   \
            <<<grid, block, 0, stream>>>(buffer, buffer_src, attr_mask, batch_size, head_num, seq_len, kv_len, scalar);        \
    }

template<typename T, typename T_IN>
void invokeMaskedSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const int* attr_mask,
                         const int batch_size,
                         const int seq_len,
                         const int kv_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream)
{

    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && kv_len % 2 == 0;
    dim3 block((kv_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        // FT_CHECK(seq_len <= 4096);
    }
}

template<typename T>
__global__ void
transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head + head_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<typename T>
void invokeTransposeQKV(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        // FT_CHECK(grid.x * seq_per_block == batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }
#ifdef ENABLE_BF16
            else {
                transpose<__nv_bfloat162><<<grid, block, 0, stream>>>(
                    (__nv_bfloat162*)src, (__nv_bfloat162*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }
#endif
        }
        else {
            block.x = seq_per_block * size_per_head;
            transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
        }
    }
    else {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}


template<typename T>
__global__ void generalLayerNorm(
    const T* __restrict input, const T* __restrict gamma, const T* __restrict beta, T* output, int m, int n)
{
    const int tid = threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float mean = 0.0f;
    float variance = 0.0f;

    float local_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += (float)(ldg(&input[blockIdx.x * n + i]));
    }

    mean = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        s_mean = mean / n;
    }
    __syncthreads();

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = (float)(ldg(&input[blockIdx.x * n + i])) - s_mean;
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / n + 1e-6f);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        float beta_val = (beta == nullptr) ? 0.0f : (float)ldg(&beta[i]);
        output[blockIdx.x * n + i] =
            (T)((((float)input[blockIdx.x * n + i] - s_mean) * s_variance) * (float)(ldg(&gamma[i])) + beta_val);
    }
}

template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            int opt_version)
{
    dim3 grid(m);
    if (false) {
        
    }
    else {
        dim3 block(min(n, 1024));

        /* For general cases, n is equal to hidden_units, e.g., 512/1024.
            Since we have warp shuffle inside the code, block.x % 32 should be 0.
        */
        if (n % 32 != 0) {
            block.x = 1024;
        }

        /* should pay attention to the rsqrt precision*/
        generalLayerNorm<T><<<grid, block, 0, stream>>>(input, gamma, beta, out, m, n);  // For gpt-3
    }
}

#undef HALF_LAYERNORM_OPT
#undef HALF_LAYERNORM_OPT2

template void invokeGeneralLayerNorm(float* out,
                                     const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
template void invokeGeneralLayerNorm(half* out,
                                     const half* input,
                                     const half* gamma,
                                     const half* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
#ifdef ENABLE_BF16
template void invokeGeneralLayerNorm(__nv_bfloat16* out,
                                     const __nv_bfloat16* input,
                                     const __nv_bfloat16* gamma,
                                     const __nv_bfloat16* beta,
                                     const int m,
                                     const int n,
                                     cudaStream_t stream,
                                     int opt_version);
#endif

template<typename T>
__global__ void addBiasResidual(T* output, const T* input, const T* bias, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index];
        output[blockIdx.x * n + col_index] =
            output[blockIdx.x * n + col_index] + input[blockIdx.x * n + col_index] + bias_val;
    }
}

template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    addBiasResidual<<<grid, block, 0, stream>>>(output, input, bias, m, n);
}

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
    printf("MHAPlugin::MHAPlugin, readFromBuffer isCrossAtten: %s \n", isCrossAtten?"true":"false");

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
    int seq_len1    = inputDesc[1].dims.d[1];    // T1 for k v 
    
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
    FILE * pFile;
    pFile = fopen(fname.c_str(),"w");

    T dst_cpu[N];
    cudaMemcpy(dst_cpu, src, N*sizeof(T), cudaMemcpyDeviceToHost);
    for(int i=0; i<N; i++)
    {
        fprintf(pFile, "%f\n", (float)(dst_cpu[i]));
    }
    fclose(pFile);
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
    const int seq_len1    = inputDesc[1].dims.d[1];    // T1 for k v
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
    // dump2Txt((float*)(query_in), batch_size*seq_len0*d_model, "dump_trt_input/query_in.txt");
    // dump2Txt((float*)(enc_in), batch_size*seq_len1*d_model, "dump_trt_input/enc_in.txt");
    // dump2Txt((int*)(enc_mask), batch_size/10, "dump_trt_input/enc_mask.txt");
    // dump2Txt((float*)(query_weight_kernel), d_model*d_model, "dump_trt_input/query_kernel.txt");
    // dump2Txt((float*)(query_weight_bias)  , d_model,         "dump_trt_input/query_bias.txt");
    // dump2Txt((float*)(key_weight_kernel), d_model*d_model, "dump_trt_input/key_kernel.txt");
    // dump2Txt((float*)(key_weight_bias)  , d_model,         "dump_trt_input/key_bias.txt");
    // dump2Txt((float*)(value_weight_kernel), d_model*d_model, "dump_trt_input/value_kernel.txt");
    // dump2Txt((float*)(value_weight_bias)  , d_model,         "dump_trt_input/value_bias.txt");
    // dump2Txt((float*)(output_weight_kernel), d_model*d_model, "dump_trt_input/linear_kernel.txt");
    // dump2Txt((float*)(output_weight_bias)  , d_model,         "dump_trt_input/linear_bias.txt");
    // printf("debug  inputs %d %d %d, %d %d %d \n", batch_size, seq_len0, d_model, 
    //                 inputDesc[1].dims.d[0], inputDesc[1].dims.d[1], inputDesc[1].dims.d[2]);
    int ws_offset  = 0;
    float* q_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* q_share = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* qk_buf  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len0*seq_len1;
    float* out_buf = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len0*d_model;
    float* k_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* k_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* v_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;
    float* v_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len1*d_model;

    cudaStreamSynchronize(stream);
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
    cudaStreamSynchronize(stream);
    // dump2Txt((float*)(query_in), batch_size*seq_len0*d_model, "dump_trt/q_in.txt");
    // dump2Txt((float*)(enc_in), batch_size*seq_len1*d_model, "dump_trt/enc_in.txt");
    // dump2Txt(q_buf, batch_size*seq_len0*d_model, "dump_trt/q_buf.txt");
    // dump2Txt(k_buf, batch_size*seq_len1*d_model, "dump_trt/k_buf.txt");
    // dump2Txt(v_buf, batch_size*seq_len1*d_model, "dump_trt/v_buf.txt");

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
    cudaStreamSynchronize(stream);
    // dump2Txt(q_share, batch_size*seq_len0*d_model, "dump_trt/q_bias_trans.txt");
    // dump2Txt(k_bias, batch_size*seq_len1*d_model, "dump_trt/k_bias_trans.txt");
    // dump2Txt(v_bias, batch_size*seq_len1*d_model, "dump_trt/v_bias_trans.txt");

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
    cudaStreamSynchronize(stream);
    // dump2Txt(qk_buf, batch_size*head_num*seq_len0*seq_len1, "dump_trt/qk_buf.txt");
    
    float scalar = 1 / sqrtf(size_per_head * 1.0f);
    invokeMaskedSoftMax(qk_buf,
                        qk_buf,
                        enc_mask,
                        batch_size,
                        seq_len0,
                        seq_len1,
                        head_num,
                        scalar,
                        stream
        );
    cudaStreamSynchronize(stream);
    // dump2Txt(qk_buf, batch_size*head_num*seq_len0*seq_len1, "dump_trt/softmax.txt");

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
    cudaStreamSynchronize(stream);
    // dump2Txt(q_buf, batch_size*head_num*seq_len0*size_per_head, "dump_trt/qkv.txt");

    invokeTransposeQKV(
        out_buf, q_buf, batch_size, seq_len0, head_num, size_per_head, stream);
    cudaStreamSynchronize(stream);
    // dump2Txt(out_buf, batch_size*head_num*seq_len0*size_per_head, "dump_trt/qkv_trans.txt");
    // dump2Txt(output_weight_kernel, d_model*d_model, "dump_trt/lw.txt");

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
    cudaStreamSynchronize(stream);
    // dump2Txt((float*)(outputs[0]), batch_size*head_num*seq_len0*size_per_head, "dump_trt/out_linear.txt");
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
    // dump2Txt((float*)(outputs[0]), batch_size*head_num*seq_len0*size_per_head, "dump_trt/out_final.txt");

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
