#!/bin/bash
cd ..
export PYTHONPATH=../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}

echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize_engine2.py \
      --config $repoPath/datasets/train.yaml \
      --test_data $repoPath/datasets/data.list \
      --data_type raw \
      --gpu 0 \
      --dict $repoPath/datasets/lang_char.txt \
      --encoder_plan $repoPath/encoder_fix_for_pass.plan \
      --decoder_plan $repoPath/decoder_fix_for_pass.plan \
      --result_file $repoPath/log/engine2_result.txt \
      --batch_size $batch \
      --mode $mode \
      --so $repoPath