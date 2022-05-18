#include "cublas.h"
#include "cublas_v2.h"

template<typename T>
__global__ void transpose_4d_batch_major_mem_q_cache(
    T* v_dst, const T* v_src, const int batch_size, const int seq_length, const int d_model);

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* qkv_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head);

template<typename T, typename T_IN>
void invokeMaskedSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const int* attr_mask,
                         const bool isCrossAtten,
                         const int batch_size,
                         const int seq_len,
                         const int kv_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream);

template<typename T>
void invokeTransposeQKV(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
void invokeAddBiasResidual(T* output, const T* input, const T* bias, const int m, const int n, cudaStream_t stream);


template<typename T>
void invokeGeneralLayerNorm(T* out,
                            const T* input,
                            const T* gamma,
                            const T* beta,
                            const int m,
                            const int n,
                            cudaStream_t stream,
                            int opt_version);

// template void invokeGeneralLayerNorm(float* out,
//                                      const float* input,
//                                      const float* gamma,
//                                      const float* beta,
//                                      const int m,
//                                      const int n,
//                                      cudaStream_t stream,
//                                      int opt_version);

// template void invokeGeneralLayerNorm(half* out,
//                                      const half* input,
//                                      const half* gamma,
//                                      const half* beta,
//                                      const int m,
//                                      const int n,
//                                      cudaStream_t stream,
//                                      int opt_version);
// #ifdef ENABLE_BF16
// template void invokeGeneralLayerNorm(__nv_bfloat16* out,
//                                      const __nv_bfloat16* input,
//                                      const __nv_bfloat16* gamma,
//                                      const __nv_bfloat16* beta,
//                                      const int m,
//                                      const int n,
//                                      cudaStream_t stream,
//                                      int opt_version);
// #endif

