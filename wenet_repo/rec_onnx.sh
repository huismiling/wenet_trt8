#!/bin/bash
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


# trtexec --onnx=./encoder_final_quant.onnx --saveEngine=./encoder.plan \
# --minShapes=speech:1x16x80,speech_lengths:1 \
# --optShapes=speech:4x64x80,speech_lengths:4 \
# --maxShapes=speech:16x256x80,speech_lengths:16 \
# --workspace=23028 --verbose --int8 \
# 2>&1 | tee log_encoder.txt


# trtexec --onnx=./decoder_final_quant.onnx --saveEngine=./decoder.plan \
# --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
# --optShapes=encoder_out:4x64x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
# --maxShapes=encoder_out:16x256x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
# --workspace=23028 --verbose --int8 \
# 2>&1 | tee log_decoder.txt

