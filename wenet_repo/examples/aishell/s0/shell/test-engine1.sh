#!/bin/bash
cd ..
export PYTHONPATH=../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
repoPath=/workspace/wenet_trt8

echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize_engine1.py \
      --config $repoPath/datasets/train.yaml \
      --test_data $repoPath/datasets/data.list \
      --data_type raw \
      --gpu 0 \
      --dict $repoPath/datasets/lang_char.txt \
      --encoder_plan $repoPath/encoder.plan \
      --decoder_plan $repoPath/decoder.plan \
      --result_file $repoPath/log/tensorrt_result2.txt \
      --batch_size $batch \
      --mode $mode \
      --so $repoPath