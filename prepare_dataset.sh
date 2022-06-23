#!/bin/bash

repoPath=/workspace/wenet_trt8
# test_data_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/test.tar
# torch_model_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/final.pt
# encoder_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/encoder.onnx
# decocer_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/decoder.onnx
# wget $test_data_url -P $repoPath/datasets/
# wget $torch_model_url -P $repoPath/model/
# wget $encoder_url -P $repoPath/model/
# wget $decocer_url -P $repoPath/model/

python prepare_dataset.py $repoPath