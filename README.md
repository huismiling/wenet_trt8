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
      
      <img src="https://user-images.githubusercontent.com/92794867/175815924-0b59e1fd-82c7-417c-a7a0-703a8b61400a.png" height="300px" div align=right />
      
      <img src="https://user-images.githubusercontent.com/92794867/175853930-4250e78b-f5f9-47c6-adda-a130508403d1.png" height="300px" />
      
   3. 针对 encoder 中两个 Slice 算子连接使用单 Slice 替换。
      
      
      <img src="https://user-images.githubusercontent.com/92794867/175815933-42cd9921-db51-4300-9a50-0f62813b0626.png" height="300px" />
      <img src="https://user-images.githubusercontent.com/92794867/175854070-48bddaba-4927-474f-8d47-9b9be2e01a01.png" height="300px" />
      
      
      
   4. 针对大矩阵和固定矩阵乘法计算连接 Slice 的情况对固定的运算进行提前计算，减少了运行时多与的计算。
      <img src="https://user-images.githubusercontent.com/92794867/175815950-202b41e9-7c51-418e-afe0-b5402e948868.png" height="300px" />
      <img src="https://user-images.githubusercontent.com/92794867/175857282-273b0fa0-02ca-4f9d-b742-dd150539e8e2.png" height="300px" />
      
   5. 对 LayerNorm 操作的大量算子使用 fp16/fp32 高效 Plugin 替换。
      <img src="https://user-images.githubusercontent.com/92794867/175815955-1b6f6283-fa1f-49e9-84e0-07aee1229feb.png" height="300px" />
      <img src="https://user-images.githubusercontent.com/92794867/175854299-1d41f089-188c-4f4c-9aaa-717cb9cb24c2.png" height="300px" />
      
   6. 针对 Attention Mask  Softmax 部分使用 AttentionMaskSoftmaxPlugin 进行替换。
      <img src="https://user-images.githubusercontent.com/92794867/175815961-d6beb1fa-1a42-4afe-9ffb-34d0404a713d.png" height="300px" />
      <img src="https://user-images.githubusercontent.com/92794867/175854399-d6c406f1-5a4e-4dea-aea0-0f9122e5c511.png" height="300px" />
      
   7. 对于所有的 mask 计算加入到输入，提前计算好根据输入的 mask，减少在运行时额外计算。
         <img src="https://user-images.githubusercontent.com/92794867/175854521-c8e2ebdb-894d-4869-8eaf-a8645fb06fa5.png" height="300px" />

   8. 根据 FastTransformer 实现上述 Plugin，实现 fp16/fp32 的模板。使用 onnxruntime 对所有 Conv/MatMul节点 weights 进行 int8 量化，对 Softmax/bias 不进行量化，对 Plugin 包含的节点进行量化。同时使用 ppq 中的 TensorRT quant 配置对 encoder decoder 全部节点进行自适应量化，对 Plugin 包含的节点选择 fp16/fp32 构建。



## 精度与加速效果
- Environment
  - TensorRT 8.4 GA
  - CUDA11.7 CUDNN 8.4.1
  - NVIDIA Telsa A10 24GB
  - RAM 32GB
  - Ubuntu 20.04
  - python3.8
  - Driver 510.47.03
  - See other environment in requirements.txt

### nsys profile

对原始模型转换 engine 分析：

<img src="https://user-images.githubusercontent.com/92794867/175858320-ad0c9475-2110-4066-b583-447fcdbf932d.png" height="300px" />

图中 Slice_84_Cast 部分总耗时较大，这部分我们采用删除 Slice_84 将两个连续的 Slice 替换为一个避免了在这部分推理的耗时。修改后的效果如下:

<img src="https://user-images.githubusercontent.com/92794867/175858672-5186e04e-5a97-44fb-bbbd-afa5cc98824f.png" height="300px" />
可以看到这部分融合后总耗时降低4倍，说明本次修改提速较大。

 <img src="https://user-images.githubusercontent.com/92794867/175858995-25a36486-1598-4449-a457-f0348551cf27.png" height="300px" />
模型中存在很多 Conv 算子，他们的耗时普遍较大，因此我们尝试将所有 Conv 的权重进行量化，从而提高 Conv 的计算速度，修改后部分结果如下:

<img src="https://user-images.githubusercontent.com/92794867/175859460-3272215b-9293-4239-90a4-e6d3f348a1c7.png" height="300px" />
Conv 经过量化后速度提升一倍，提速效果较好。同样的我们对于 MatMul 节点也进行了量化。

我们测试对比了 Plugin 的推理速度

<img src="https://user-images.githubusercontent.com/92794867/175861573-a7c96f1a-8de0-4db9-a95e-066c76051cdb.png" height="50px" />
<img src="https://user-images.githubusercontent.com/92794867/175860696-2fd7fee7-8948-4230-8ada-c68a77ae1ed5.png" height="50px" />
TensorRT 自动融合会将 masked softmax 部分操作融合到一起，使用 Plugin 后可以对这部分操作加速。

| model             | b1(ms)  | b4(ms)  | b8(ms)  | b16(ms) | error b1 | error b4 | error b8 | error b16 |
| ----------------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- | --------- |
| original encoder  | 18.4037 | 23.5395 | 26.1690 | 30.7799 |          |          |          |           |
| original decoder  | 20.8410 | 22.3583 | 22.9112 | 23.6168 | 4.63     | 4.78     | 4.82     | 4.85      |
| ort quant encoder | 7.8731  | 11.3805 | 5.6806  | 26.1500 |          |          |          |           |
| ort quant decoder | 3.5666  | 9.2714  | 17.830  | 36.3769 | 6.06     | 6.25     | 6.43     | 6.72      |
| ppq quant encoder | 12.1374 | 29.8226 | 29.7251 | 48.8531 |          |          |          |           |
| ppq quant decoder | 2.6780  | 8.5099  | 16.4860 | 33.0441 | 5.71     | 6.17     | 6.70     | 7.29      |

本次模型评价指标没有选择与初赛类似的方式，原因是 encoder 的输出包含范围较大的整数和浮点数，输出经过 decoder 后会经过后续的解码，中间输出对 decoder 的影响较小，当模型经过量化后，使用相对误差和绝对误差评判结果，误差可能达到较大的量级，但是对于语音识别任务而言，效果影响很小，经过测试量化后的模型虽然误差超过10%，但是识别错误率仅仅提高1.5%左右。

original  是经过 pytorch 导出的 onnx 直接使用 trtexec 导出 engine 然后分别对 batch=1/4/8/16 下的推理速度和识别错误率。

ort quant 是我们使用 onnxruntime 提供的 int8 量化对 encoder 和 decoder 中 Conv MatMul 部分量化，然后替换 Plugin 生成 engine 然后分别对 batch=1/4/8/16 下的推理速度和识别错误率。

ppq quant 是我们使用 [PPQ](https://github.com/openppl-public/ppq) 提供的 int8 量化对 encoder 和 decoder 中 Conv MatMul 部分量化，然后替换 Plugin 生成 engine 然后分别对 batch=1/4/8/16 下的推理速度和识别错误率。

上述结果表明，在较小的 batch(1/4/8) 下，经过量化后的模型 encoder 提速两倍左右，decoder 提速五倍左右，错误率仅仅提高 1.5%。

语音识别任务对于端测部署常常使用 batch=1 进行推理，因此我们提供的量化和 Plugin 解决方案完全可以满足工业使用要求，并且提速明显，识别实时性强。

除此之外，对于 batch=16 的情况下，我们发现量化后的模型效果可能变差，分析如下:

<img src="https://user-images.githubusercontent.com/92794867/175862575-234168e1-9560-4872-b8f8-ce5d56c6c431.png" height="50px" />
<img src="https://user-images.githubusercontent.com/92794867/175862586-03b0be6f-3fd4-421e-bab7-11bb16b99117.png" height="50px" />
为了保持模型精度，我们只做了部分量化，TensorRT 会针对每个量化的节点插入 QDQ 的计算，由于 encoder 和 decoder 中大量的算子被量化，因此这部分转化节点在 batch 很大的情况下会有影响。

## Bug报告（可选）




- Environment

  - TensorRT 8.4 GA
  - CUDA11.7 CUDNN 8.4.1
  - NVIDIA Telsa A10 24GB
  - RAM 32GB
  - Ubuntu 20.04
  - python3.8
  - Driver 510.47.03
  - See other environment in requirements.txt

- Reproduction Steps
  - bug 1:

    ``` shell
    # download onnx
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/bug_report/encoder_group_conv_quant.onnx
    # build engine will show group conv201 error
    trtexec --onnx=./encoder_group_conv_quant.onnx --saveEngine=./encoder.plan \
            --minShapes=speech:1x1x80,speech_lengths:1 \
            --optShapes=speech:4x750x80,speech_lengths:4 \
            --maxShapes=speech:16x1500x80,speech_lengths:16 \
            --workspace=8192 --int8 --verbose 2>&1 | tee ./log.log
    # log
    [06/27/2022-14:12:39] [E] Error[10]: [optimizer.cpp::computeCosts::3628] Error Code 10: Internal Error (Could not find any implementation for node 3161 + PPQ_Operation_102 + (Unnamed Layer* 1958) [Shuffle] + Conv_201 + PWN(Sigmoid_202, Mul_203).)
    [06/27/2022-14:12:39] [E] Error[2]: [builder.cpp::buildSerializedNetwork::636] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
    ```

  - bug 2:

    ``` shell
    # download onnx
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/bug_report/encoder_replaced_flatten.onnx
    # build plugin
    sh build_plugin.sh
    # build engine will show matmul61 error
    trtexec --onnx=./encoder_replaced_flatten.onnx --saveEngine=./encoder.plan \
            --minShapes=speech:1x1x80,speech_lengths:1,speech_lengths_mask:1x40x40 \
            --optShapes=speech:4x750x80,speech_lengths:4,speech_lengths_mask:4x220x220 \
            --maxShapes=speech:16x1500x80,speech_lengths:16,speech_lengths_mask:16x400x400 \
            --plugins=./libwenet_plugin.so --int8 \
            --workspace=24576 --verbose 2>&1 | tee ./log/encoder_build.log
    # log
    [06/27/2022-06:42:35] [E] Error[2]: [qdqGraphOptimizer.cpp::reportWeightlessTwoInputConvolutionAsError::230] Error Code 2: Internal Error (MatMul_61: Could not fuse 2nd input (kernel weights) of CONVOLUTION)
    [06/27/2022-06:42:35] [E] Error[2]: [builder.cpp::buildSerializedNetwork::636] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
    ```

  - bug 3: 

    ``` shell
    # download onnx
    wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/bug_report/encoder_quant_pwn_fusion.onnx
    # build engine will show group conv213 error
    trtexec --onnx=./encoder_quant_pwn_fusion.onnx --saveEngine=./encoder.plan \
            --minShapes=speech:1x1x80,speech_lengths:1 \
            --optShapes=speech:4x750x80,speech_lengths:4 \
            --maxShapes=speech:16x1500x80,speech_lengths:16 \
            --workspace=8192 --int8 --verbose 2>&1 | tee ./log.log
    # log
    [06/27/2022-14:56:04] [E] Error[10]: [optimizer.cpp::computeCosts::3628] Error Code 10: Internal Error (Could not find any implementation for node onnx::Conv_3304 + PPQ_Operation_58 + (Unnamed Layer* 2015) [Shuffle] + Conv_213 + PWN(Sigmoid_214, Mul_215).)
    [06/27/2022-14:56:04] [E] Error[2]: [builder.cpp::buildSerializedNetwork::636] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
    ```

    

  - bug 4: (8.4.1.4 bug, have been fixed)
- Expected Behavior
  - 1: group conv 应该可以支持 int8 量化（特殊形状的 kernel 也应该支持）
  - 2: torch.view 换成 flatten 导出的 onnx Reshape -1 位置应该是固定值，应该可以支持 shape 推导
  - 3: ppq 量化后的 conv 应该可以正常融合
  - 4: TRT 8.4.1.4 发现的 bug，TRT8.4.1.5 GA 已经修复了
- Actual Behavior

  - 1: group conv 量化不支持或特殊形状 kernel 不支持
  - 2: MatMul_61 节点因为 reshape -1 的操作无法是别是固定 shape 的 weights
  - 3: 量化后的 conv213 不支持融合
- Additional Notes
  - Provide any additional context here you think might be useful for the TensorRT team to help debug this issue (such as experiments done, potential things to investigate).




## 经验与体会（可选）

