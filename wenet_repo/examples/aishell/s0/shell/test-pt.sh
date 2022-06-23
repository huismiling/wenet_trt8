#!/bin/bash
cd ..
export PYTHONPATH=../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
repoPath=/home/ubuntu/study/repo/wenet_trt8

echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize.py \
     --gpu 0 \
     --mode $mode \
     --config $repoPath/datasets/train.yaml \
     --data_type raw \
     --test_data $repoPath/datasets/data.list \
     --checkpoint $repoPath/model/final.pt \
     --beam_size 10 \
     --batch_size $batch \
     --penalty 0.0 \
     --dict $repoPath/datasets/lang_char.txt \
     --ctc_weight 0.3 \
     --reverse_weight 0.0 \
     --result_file $repoPath/log/$mode