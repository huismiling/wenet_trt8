#!/bin/bash
gpuGB=24
workspace=$(($gpuGB * 1024))
gpuStr="$gpuGB""G"
log=1
python main.py --name all --dynamic --sim --ln --skip
cd /target/LayerNorm && make clean && make && cp LayerNorm.so ..
cd /target/SkipLayerNorm && make clean && make && cp SkipLayerNorm.so ..
cd /workspace
echo "GPU MEM is $gpuStr and $workspace MB"
echo "log is $log"

if [[ $log == 1 ]]; then
  echo "Save verbose logs to ./log/encoder.txt and ./log/decoder.txt"
  echo "**************************************************"

  echo "Start convert encoder.onnx to encoder.plan"

#  trtexec --onnx=/target/encoder.onnx --saveEngine=/target/encoder.plan \
#    --minShapes=speech:1x16x80,speech_lengths:1 \
#    --optShapes=speech:4x64x80,speech_lengths:4 \
#    --maxShapes=speech:16x256x80,speech_lengths:16 \
#    --workspace=$workspace --verbose --fp16 \
#    --plugins=/target/SkipLayerNorm.so \
#    2>&1 | tee >/target/log/encoder.txt

  python /target/buildEngine.py --name encoder --soFile /target/SkipLayerNorm.so 2>&1 --fp16 | tee >/target/log/encoder.txt

  echo "Finish convert encoder.onnx to encoder.plan"

  echo "Start convert decoder.onnx to decoder.plan"

#   trtexec --onnx=/target/decoder.onnx --saveEngine=/target/decoder.plan \
#     --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
#     --optShapes=encoder_out:4x64x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
#     --maxShapes=encoder_out:16x256x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
#     --workspace=$workspace --verbose --noTF32 --fp16 \
#     --plugins=/target/SkipLayerNorm.so \
#     2>&1 | tee >/target/log/decoder.txt

  python /target/buildEngine.py --name decoder --soFile /target/SkipLayerNorm.so 2>&1 --fp16 | tee >/target/log/decoder.txt

  echo "Finish convert decoder.onnx to decoder.plan"

  echo "Finish convert all *onnx to all *plan"
  echo "**************************************************"

else
  echo "Do not save verbose logs"

  trtexec --onnx=/target/encoder.onnx --saveEngine=/target/encoder.plan \
    --minShapes=speech:1x16x80,speech_lengths:1 \
    --optShapes=speech:32x64x80,speech_lengths:32 \
    --maxShapes=speech:64x256x80,speech_lengths:64 \
    --workspace=$workspace --fp16 \
    --plugins=/target/LayerNorm.so

  echo "Finish convert encoder.onnx to encoder.plan"

  trtexec --onnx=/target/decoder.onnx --saveEngine=/target/decoder.plan \
    --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
    --optShapes=encoder_out:32x64x256,encoder_out_lens:32,hyps_pad_sos_eos:32x10x64,hyps_lens_sos:32x10,ctc_score:32x10 \
    --maxShapes=encoder_out:64x256x256,encoder_out_lens:64,hyps_pad_sos_eos:64x10x64,hyps_lens_sos:64x10,ctc_score:64x10 \
    --workspace=$workspace --noTF32 --fp16 \
    --plugins=/target/LayerNorm.so

  echo "Finish convert decoder.onnx to decoder.plan"
fi
python /workspace/testEncoderAndDecoder.py
