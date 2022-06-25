#!/bin/bash
cd ..
export PYTHONPATH=../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}

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
      --result_file $repoPath/log/engine1_result.txt \
      --batch_size $batch \
      --mode $mode \
      --so $repoPath

python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      $repoPath/datasets/text \
      $repoPath/log/engine1_result.txt \
      2>&1 | tee $repoPath/log/wer/engine1.txt

python wenet/bin/compute-speed.py \
      $repoPath/log/npys \
      engine1 \
      $mode

