#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T>
__global__ void layerNormKernel(T *pInput, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    __shared__ T temp[128];

    T value0 = pInput[index];
    T value1 = pInput[index + 128];

    temp[tx] = value0 + value1;
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    T mean = temp[0] /(T)256;
    __syncthreads();

    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    T var = temp[0] / (T)256;

    pOutput[index]       = (value0 - mean) * (T)rsqrtf(var + (T)1e-5);
    pOutput[index + 128] = (value1 - mean) * (T)rsqrtf(var + (T)1e-5);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    
    if (inputDesc[0].type == DataType::kFLOAT)
    {

        (layerNormKernel<float>)<<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);

    }
    else
    {

        (layerNormKernel<half>)<<<nBlock, 128, 0, stream>>>((half *)inputs[0], (half *)outputs[0]);

        
    }
    
    return 0;

}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

