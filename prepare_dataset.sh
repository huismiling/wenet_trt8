#!/bin/bash

test_data_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/test.tar
torch_model_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/final.pt
encoder_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/encoder.onnx
decocer_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/decoder.onnx
npys_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/npys.tar
data_url=https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/tensorrt/data.tar.gz
wget $test_data_url -P $repoPath/datasets/
wget $npys_url -P $repoPath/datasets/
wget $torch_model_url -P $repoPath/model/
wget $encoder_url -P $repoPath/model/
wget $decocer_url -P $repoPath/model/
wget $data_url -P $repoPath/datasets/

python prepare_dataset.py $repoPath
cd ./datasets/
tar -xvf npys.tar
tar -zxvf data.tar.gz
mv data ort_quant_data