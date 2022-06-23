#!/bin/bash
# 1468 366 366 39
python3 ./recognize_onnx.py \
--config ./20210204_conformer_exp/train.yaml \
--test_data ./data/test/data.list \
--data_type raw \
--gpu 0 \
--dict data/dict/lang_char.txt \
--encoder_onnx ./onnx_export/encoder.onnx \
--decoder_onnx ./onnx_export/decoder.onnx \
--result_file ./result_text \
--batch_size 16 \
--mode attention_rescoring