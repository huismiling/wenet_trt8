
set -v 
set -e 
# python onnxsim.py encoder_new.onnx encoder_sim.onnx 4  \
#     --dynamic-input-shape --input-shape "speech:16,256,80" "speech_lengths:16" 

# python onnxsim.py decoder.onnx decoder_sim.onnx 4  \
#     --dynamic-input-shape --input-shape "encoder_out:16,80,256" "encoder_out_lens:16" "hyps_pad_sos_eos:16,10,64" "hyps_lens_sos:16,10" "ctc_score:16,10"

# trtexec --workspace=3000 \
#         --onnx=./encoder_new.onnx \
#         --optShapes=speech:4x64x80,speech_lengths:4 \
#         --maxShapes=speech:16x256x80,speech_lengths:16 \
#         --minShapes=speech:1x16x80,speech_lengths:1 \
#         --shapes=speech:1x16x80,speech_lengths:1 \
#         --saveEngine=./encoder.plan \
#         --verbose

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# mkdir wenet_plugin/build/
# cd wenet_plugin/build/
# cmake ..
# make -j
# cd -
# cp wenet_plugin/build/libmhalugin.so . -s

python fix_decoder.py
python onnx_opt.py decoder decoder_new.onnx decoder_quant.onnx
python fix_quant_model.py decoder_quant.onnx decoder_quant_fixed.onnx
python replace_decoder_attn.py decoder_quant_fixed.onnx decoder_replaced.onnx
python onnx2trt.py decoder decoder_replaced.onnx decoder.plan 2>&1 | tee build_decoder.log


# trtexec --workspace=12000 \
#         --onnx=./decoder_sim.onnx \
#         --optShapes=encoder_out:4x64x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
#         --maxShapes=encoder_out:16x256x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
#         --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
#         --shapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
#         --saveEngine=./decoder.plan \
#         --verbose


