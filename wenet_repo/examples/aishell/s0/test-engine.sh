#!/bin/bash
export PYTHONPATH=../../../:$PYTHONPATH

batch=${1:-1}
mode=${2:-attention_rescoring}
echo '**************************************************'
echo "mode is $mode\nbatch is $batch"
echo '**************************************************'

python recognize_engine.py \
      --config 20210204_conformer_exp/train.yaml \
      --test_data data/test/data.list \
      --data_type raw \
      --gpu 0 \
      --dict data/dict/lang_char.txt \
      --encoder_plan /workspace/work/wenet_trt8/encoder.plan \
      --decoder_plan /workspace/work/wenet_trt8/decoder.plan \
      --result_file onnx_export/onnx_result.txt \
      --batch_size $batch \
      --mode $mode \
      --so /workspace/work/wenet_trt8/