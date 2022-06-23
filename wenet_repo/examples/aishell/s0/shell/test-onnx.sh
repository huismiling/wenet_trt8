#!/bin/bash
cd ..
export PYTHONPATH=../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
repoPath=/home/ubuntu/study/repo/wenet_trt8

echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize_onnx.py \
      --config $repoPath/datasets/train.yaml \
      --test_data $repoPath/datasets/data.list \
      --data_type raw \
      --gpu 0 \
      --dict $repoPath/datasets/lang_char.txt \
      --encoder_onnx $repoPath/model/encoder.onnx \
      --decoder_onnx $repoPath/model/decoder.onnx \
      --result_file $repoPath/log/onnx_result.txt \
      --batch_size $batch \
      --mode $mode


