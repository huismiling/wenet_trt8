#!/bin/bash
set -x
workspace=24576
python3 fix2pass.py

trtexec --onnx=./encoder_fix_for_pass.onnx --saveEngine=./encoder_fix_for_pass.plan \
        --minShapes=speech:1x1x80,speech_lengths:1 \
        --optShapes=speech:4x750x80,speech_lengths:4 \
        --maxShapes=speech:16x1500x80,speech_lengths:16 \
        --workspace=$workspace --verbose 2>&1 | tee ./log/encoder_noopt.log

trtexec --onnx=./model/decoder.onnx --saveEngine=./decoder_fix_for_pass.plan \
        --minShapes=encoder_out:1x40x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
        --optShapes=encoder_out:4x165x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
        --maxShapes=encoder_out:16x370x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
        --workspace=$workspace --verbose 2>&1 | tee ./log/decoder_noopt.log

# # nsys show 
# trtexec --onnx=./encoder_fix_for_pass.onnx --saveEngine=./encoder_fix_for_pass.plan \
#         --minShapes=speech:1x16x80,speech_lengths:1 \
#         --optShapes=speech:4x64x80,speech_lengths:4 \
#         --maxShapes=speech:16x256x80,speech_lengths:16 \
#         --workspace=$workspace --verbose 2>&1 | tee ./log/encoder_noopt.log

# trtexec --onnx=./model/decoder.onnx --saveEngine=./decoder_fix_for_pass.plan \
#         --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
#         --optShapes=encoder_out:4x64x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
#         --maxShapes=encoder_out:16x256x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
#         --workspace=$workspace --verbose 2>&1 | tee ./log/decoder_noopt.log




# trtexec --onnx=./encoder_fix_for_pass.onnx --saveEngine=./encoder_fix_for_pass.plan \
#         --minShapes=speech:1x1x80,speech_lengths:1 \
#         --optShapes=speech:4x750x80,speech_lengths:4 \
#         --maxShapes=speech:16x1500x80,speech_lengths:16 \
#         --workspace=24576 --verbose 2>&1 | tee /workspace/encoder.log