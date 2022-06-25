#!/bin/bash

set -x
workspace=24576
#git submodule update --init --recursive
git clone https://github.com/huismiling/FasterTransformer_wenet.git FasterTransformer_wenet
cd ./other_branch/
tar -xvf ppq.tar
cd ./ppq/
pip3 install -v -e .
cd ./quant_ppq/

python3 encoder_fix.py
python3 encoder_quant.py
python3 model_dynamic.py encoder
python3 replace_encoder_ln.py

#python3 decoder_fix.py
python3 decoder_quant.py
python3 model_dynamic.py decoder
python3 replace_decoder_ln.py

mv ./encoder_replace.onnx ../../encoder_replace.onnx
mv ./decoder_replace.onnx ../../decoder_replace.onnx

cd ../../
trtexec --onnx=./encoder_replace.onnx --saveEngine=./encoder.plan \
        --minShapes=speech:1x1x80,speech_lengths:1 \
        --optShapes=speech:4x750x80,speech_lengths:4 \
        --maxShapes=speech:16x1500x80,speech_lengths:16 \
        --plugins=./libwenet_plugin.so \
        --workspace=$workspace --verbose 2>&1 | tee ./log/encoder_build.log

trtexec --onnx=./decoder_replace.onnx --saveEngine=./decoder.plan \
        --minShapes=encoder_out:1x40x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
        --optShapes=encoder_out:4x165x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
        --maxShapes=encoder_out:16x370x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
        --plugins=./libwenet_plugin.so \
        --workspace=$workspace --verbose 2>&1 | tee ./log/decoder_build.log

