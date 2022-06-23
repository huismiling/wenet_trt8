#!/bin/bash
export PYTHONPATH=../../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python wenet/bin/recognize_onnx.py \
      --config 20210204_conformer_exp/train.yaml \
      --test_data data/test/data.list \
      --data_type raw \
      --gpu 0 \
      --dict data/dict/lang_char.txt \
      --encoder_onnx onnx_export/encoder.onnx \
      --decoder_onnx onnx_export/decoder.onnx \
      --result_file onnx_export/onnx_result.txt \
      --batch_size $batch \
      --mode $mode


