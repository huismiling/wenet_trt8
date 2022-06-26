## 总述

本项目使用TRT8部署开源语音识别工具包[WeNet](https://github.com/wenet-e2e/wenet)。为语音识别模型在TRT8上部署提供参考方案。
原始模型来自[WeNet预训练模型](https://wenet.org.cn/wenet/pretrained_models.html）。

本项目针对ONNX模型进行优化，使用pytorch直接导出的FP32模型，没有进行量化训练，使用后量化技术对模型进行INT8量化。项目主要贡献是使用两种量化方法：PPQ和ORT(ONNX Run Time)量化对WeNet模型进行量化，在速度和耗时上两种方法有不同的表现，具体见下文。

优化效果：
* 原始ONNX模型在数据集AiShell上，测试结果WER：4.6%，耗时：Encoder 18.40ms，Decoder 20.84ms。
* 使用ORT量化后，模型测试结果WER：6.06%，耗时：Encoder 7.34ms，Decoder 3.32ms。数据集效果下降1.46%，而Encoder加速2.5倍，Decoder加速6.3倍。


Docker运行方法：
```bash
# 初始化仓库
git clone https://github.com/huismiling/wenet_trt8.git
cd wenet_trt8/
git submodule update --init

# 运行docker
docker run --gpus all -idt --name wenet_trt8 -v $PWD:/target/ registry.cn-hangzhou.aliyuncs.com/huismiling/wenet_trt8 bash
docker exec -it wenet_trt8 bash

# 在docker中进行模型转换和测试
## ORT 量化测试流程
# 0. 准备数据和模型
cd /target/
source set_env.sh
sh prepare_dataset.sh

# 1. build plugin
./build_plugin.sh

# 2. 使用ONNX Run Time进行量化，并转成TensorRT模型
./build_ort.sh

# 3. 测试TensorRT模型在数据集上的效果和耗时
cd wenet_repo/work_dir/shell/
sh test-engine1.sh

## ppq 量化测试流程
# 0. ppq quant and export
cd /target/
source set_env.sh
sh prepare_dataset.sh
sh build_ppq.sh

# 1.测试TensorRT模型在数据集上的效果和耗时
cd wenet_repo/work_dir/shell/
sh test-engine1.sh

```

## 原始模型
### 模型简介
WeNet 是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务，其主要特点如下：
* 使用 conformer 网络结构和 CTC/attention loss 联合优化方法，统一的流式/非流式语音识别方案，具有业界一流的识别效果。
* 提供云上和端上直接部署的方案，最小化模型训练和产品落地之间的工程工作。
* 框架简洁，模型训练部分完全基于 pytorch 生态，不依赖于 kaldi 等复杂的工具。
* 详细的注释和文档，非常适合用于学习端到端语音识别的基础知识和实现细节。
* 支持时间戳，对齐，端点检测，语言模型等相关功能。

本项目的模型使用预训练模型导出onnx，然后进行TRT部署。预训练模型方法导出参考[WeNet手册](https://wenet.org.cn/wenet/tutorial_aishell.html)。

训练等相关信息请参考官方：https://github.com/wenet-e2e/wenet。

### 模型优化的难点

WeNet模型分为encoder和decoder两个部分。
其中，encoder主要使用了conv和self-attention结构，而decoder使用了self-attention和cross-attention结构。

在模型转换和使用过程中存在以下问题：
* 由于是pytorch导出onnx模型，因此onnx模型中使用了大量小算子拼凑出attention功能。
* 在使用trtexec直接解析decoder模型时，在RTX 3080Ti 12G显卡上会出现显存不足的错误。
* 使用Half数据类型进行推理，encoder和decoder的精度损失严重。

针对以上问题，本项目采用以下方法进行模型优化。
* 合并onnx模型中的小算子，使用AttnMaskedSoftmax、LayerNorm等大算子替代原始小算子。
* 使用trtexec解析替换大算子的模型。
* 分析各个节点输出，定位误差大的节点，并使用高精度进行计算。
* 模型量化，使用INT8进行推理，保证精度的情况下，进一步加快速度。

## 优化过程

在拿到WeNet模型的ONNX文件后，首先，先了解一下模型的基本原理，确定模型中有没有使用特殊算子。
然后，尝试使用trtexec直接转换onnx模型。同时准备测试代码和测试数据。

经过对WeNet模型的分析，模型包含encoder和decoder两个部分。

encoder主要使用到的算子是conv、transformer等结构。

decoder使用到的算子主要就是transformer结构。

transformer结构中，使用到LayerNorm算子，但是由于onnx模型中使用小算子拼接LayerNorm，计算效率低。因此，这里是一个优化点，可以使用Plugin实现一个LayerNorm算子。

transformer中的attention结构，本可以用一个大算子来实现，但是由于项目中使用了INT8量化，量化算子主要是Conv、MatMul等。INT8量化的attention算子Plugin工程量太大，然而项目比赛时间短，因此采用了折中的方法，把attention中masked+softmax部分使用plugin算子实现，而attention中MatMul算子使用INT8量化。MaskedSoftmax算子使用高精度数据计算，支持FP16和FP32模式。这样可以保证算子精度。

确定了开发方法之后，了解到 [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) 已经实现了transformer结构的算子，因此，本项目基于FasterTransformer 实现了 WeNet 模型用到的 Plugin，仓库见[FasterTransformer_wenet](https://github.com/huismiling/FasterTransformer_wenet)。WeNet Plugin代码目录 [FasterTransformer_wenet/src/fastertransformer/tensorrt_plugin/wenet](https://github.com/huismiling/FasterTransformer_wenet/tree/main/src/fastertransformer/tensorrt_plugin/wenet)。具体编译和使用方法参见 Docker 使用方法。


优化技术总结如下：
   1. 针对 pytorch 导出的 encoder Slice 算子不支持 bool 输出进行了输入输出的强制转换。(与初赛一致)
   2. 针对 encoder 中由 torch.view 算子导出的多个琐碎算子进行了整合，使用 Flatten Transpose 算子进行替换，大大减少了琐碎算子计算。
   ![image-20220626185714687](https://user-images.githubusercontent.com/92794867/175815924-0b59e1fd-82c7-417c-a7a0-703a8b61400a.png)
   3. 针对 encoder 中两个 Slice 算子连接使用单 Slice 替换。
   ![image-20220626185753417](https://user-images.githubusercontent.com/92794867/175815933-42cd9921-db51-4300-9a50-0f62813b0626.png)
   4. 针对大矩阵和固定矩阵乘法计算连接 Slice 的情况对固定的运算进行提前计算，减少了运行时多与的计算。
   ![image-20220626185835504](https://user-images.githubusercontent.com/92794867/175815950-202b41e9-7c51-418e-afe0-b5402e948868.png)
   5. 对 LayerNorm 操作的大量算子使用 fp16/fp32 高效 Plugin 替换。
   ![image-20220626190114375](https://user-images.githubusercontent.com/92794867/175815955-1b6f6283-fa1f-49e9-84e0-07aee1229feb.png)
   6. 针对 Attention Mask  Softmax 部分使用 AttentionMaskSoftmaxPlugin 进行替换。
   ![image-20220626190530810](https://user-images.githubusercontent.com/92794867/175815961-d6beb1fa-1a42-4afe-9ffb-34d0404a713d.png)
   7. 对于所有的 mask 计算加入到输入，提前计算好根据输入的 mask，减少在运行时额外计算。
   8. 根据 FastTransformer 实现上述 Plugin，实现 fp16/fp32 的模板。使用 onnxruntime 对所有 Conv/MatMul节点 weights 进行 int8 量化，对 Softmax/bias 不进行量化，对 Plugin 包含的节点进行量化。同时使用 ppq 中的 TensorRT quant 配置对 encoder decoder 全部节点进行自适应量化，对 Plugin 包含的节点选择 fp16/fp32 构建。



## 精度与加速效果
- Environment
  - TensorRT 8.4 GA
  - CUDA11.7 CUDNN 8.4.1
  - nvcr.io/nvidia/tensorrt:22.05-py3
  - 510.47.03

| model            | b1(ms)             | b4(ms)             | b8(ms)             | b16(ms)            | error b1 | error b4 | error b8 | error b16 |
| ---------------- | ------------------ | ------------------ | ------------------ | ------------------ | -------- | -------- | -------- | --------- |
| original encoder | 18.403791837792642 | 23.53952972742475  | 26.169091756967667 | 30.779932178173716 |          |          |          |           |
| original decoder | 20.84104462876254  | 22.358369596432553 | 22.91126618394649  | 23.61681683741648  | 4.63     | 4.78     | 4.82     | 4.85      |
| our encoder      | 7.8731064322742474 | 11.380562555183946 | 15.680679668896321 | 26.150049973273937 |          |          |          |           |
| our decoder      | 3.5666254789576364 | 9.271438920847269  | 17.830058614269788 | 36.37694616035635  | 6.06     | 6.25     | 6.43     | 6.72      |

## Bug报告（可选）


- Environment
  - TensorRT 8.4 GA
  - CUDA11.7 CUDNN 8.4.1
  - nvcr.io/nvidia/tensorrt:22.05-py3
  - 510.47.03
- Reproduction Steps
  - Provide detailed reproduction steps for the issue here, including any commands run on the command line.
- Expected Behavior
  - Provide a brief summary of the expected behavior of the software. Provide output files or examples if possible.
- Actual Behavior
  - Describe the actual behavior of the software and how it deviates from the expected behavior. Provide output files or examples if possible.
- Additional Notes
  - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).

## 经验与体会（可选）

