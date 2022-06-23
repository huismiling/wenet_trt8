#!/bin/bash
export PYTHONPATH=../../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize.py \
     --gpu 0 \
     --mode $mode \
     --config 20210204_conformer_exp/train.yaml \
     --data_type raw \
     --test_data data/test/data.list \
     --checkpoint 20210204_conformer_exp/final.pt \
     --beam_size 10 \
     --batch_size $batch \
     --penalty 0.0 \
     --dict data/dict/lang_char.txt \
     --ctc_weight 0.3 \
     --reverse_weight 0.0 \
     --result_file result/$mode/text