

set -e 
set -v 

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

# mkdir /target/wenet_plugin/build/
# cd /target/wenet_plugin/build/
# cmake ..
# make -j
# cd /target/
# cp wenet_plugin/build/libmhalugin.so . -s


python fix_encoder.py
python onnx_opt.py encoder encoder_new.onnx encoder_quant.onnx
python fix_quant_model.py encoder_quant.onnx encoder_quant_fixed.onnx
python replace_encoder_attn.py encoder_quant_fixed.onnx encoder_replaced.onnx
python onnx2trt.py encoder encoder_replaced.onnx encoder.plan

# trtexec --workspace=3000 \
#         --onnx=./encoder_replaced.onnx \
#         --optShapes=speech:4x64x80,speech_lengths:4 \
#         --maxShapes=speech:16x256x80,speech_lengths:16 \
#         --minShapes=speech:1x16x80,speech_lengths:1 \
#         --shapes=speech:1x16x80,speech_lengths:1 \
#         --saveEngine=./encoder.plan \
#         --verbose \
#         --int8 \
#         --plugins=./libmhalugin.so \
#         2>&1 |tee log_build
