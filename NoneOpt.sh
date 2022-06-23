#!/bin/bash
set -x
workspace=24576
python3 fix2pass.py

trtexec --onnx=./encoder_fix_for_pass.onnx --saveEngine=./encoder_fix_for_pass.plan \
        --minShapes=speech:1x1x80,speech_lengths:1 \
        --optShapes=speech:4x800x80,speech_lengths:4 \
        --maxShapes=speech:16x1600x80,speech_lengths:16 \
        --workspace=$workspace

trtexec --onnx=./model/decoder.onnx --saveEngine=./decoder_fix_for_pass.plan \
        --minShapes=encoder_out:1x1x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x1,hyps_lens_sos:1x10,ctc_score:1x10 \
        --optShapes=encoder_out:4x800x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x32,hyps_lens_sos:4x10,ctc_score:4x10 \
        --maxShapes=encoder_out:16x1600x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
        --workspace=$workspace