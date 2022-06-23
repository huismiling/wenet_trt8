polygraphy run /target/encoder.onnx --onnxrt --trt --workspace $gpuStr --save-engine=/target/encoder.plan \
    --atol 1e-3 --rtol 1e-3 --verbose --gen-script "/target/enpoly.py" \
    --trt-min-shapes speech:[1,16,80] speech_lengths:[1] \
    --trt-opt-shapes speech:[32,64,80] speech_lengths:[32] \
    --trt-max-shapes speech:[64,256,80] speech_lengths:[64] \
    --input-shapes speech:[32,64,80] speech_lengths:[32] \
    --obey-precision-constraints

python /target/enpoly.py 2>&1 | tee >/target/log_1.txt


#  echo "Start convert all *onnx to all *plan"
  #  python /target/trt_builder.py --name encoder --workspace $gpuGB 2>&1 | tee >/target/log_all.txt
#  echo "Finish convert all *onnx to all *plan"

#    python /target/trt_builder.py --name decoder 2>&1 | tee >/target/log_2.txt

#  polygraphy run /target/decoder.onnx --onnxrt --trt --workspace $gpuStr --save-engine=/target/decoder.plan \
#    --atol 1e-3 --rtol 1e-3 --verbose --gen-script "/target/depoly.py" \
#    --trt-min-shapes encoder_out:[1,16,256] encoder_out_lens:[1] hyps_pad_sos_eos:[1,10,64] hyps_lens_sos:[1,10] ctc_score:[1,10] \
#    --trt-opt-shapes encoder_out:[32,16,256] encoder_out_lens:[32] hyps_pad_sos_eos:[32,10,64] hyps_lens_sos:[32,10] ctc_score:[32,10] \
#    --trt-max-shapes encoder_out:[64,256,256] encoder_out_lens:[64] hyps_pad_sos_eos:[64,10,64] hyps_lens_sos:[64,10] ctc_score:[64,10] \
#    --input-shapes encoder_out:[32,16,256] encoder_out_lens:[32] hyps_pad_sos_eos:[32,10,64] hyps_lens_sos:[32,10] ctc_score:[32,10] \
#    --obey-precision-constraints 2>&1 | tee >/target/log_2.txt

#  echo "Finish generate deploy.py"
#  python /target/depoly.py 2>&1 | tee >/target/log_2.txt

#  echo "Finish convert decoder.onnx to decoder.plan"


  polygraphy run /target/encoder.onnx --onnxrt --trt --workspace $gpuStr --save-engine=/target/encoder.plan \
    --atol 1e-3 --rtol 1e-3 --verbose --gen-script "/target/enpoly.py" \
    --trt-min-shapes speech:[1,16,80] speech_lengths:[1] \
    --trt-opt-shapes speech:[32,64,80] speech_lengths:[32] \
    --trt-max-shapes speech:[64,256,80] speech_lengths:[64] \
    --input-shapes speech:[32,64,80] speech_lengths:[32] \
    --obey-precision-constraints
  python /target/enpoly.py


    polygraphy run /target/decoder.onnx --onnxrt --trt --workspace $gpuStr --save-engine=/target/decoder.plan \
    --atol 1e-3 --rtol 1e-3 --verbose --gen-script "/target/depoly.py" \
    --trt-min-shapes encoder_out:[1,16,256] encoder_out_lens:[1] hyps_pad_sos_eos:[1,10,64] hyps_lens_sos:[1,10] ctc_score:[1,10] \
    --trt-opt-shapes encoder_out:[32,16,256] encoder_out_lens:[32] hyps_pad_sos_eos:[32,10,64] hyps_lens_sos:[32,10] ctc_score:[32,10] \
    --trt-max-shapes encoder_out:[64,256,256] encoder_out_lens:[64] hyps_pad_sos_eos:[64,10,64] hyps_lens_sos:[64,10] ctc_score:[64,10] \
    --input-shapes encoder_out:[32,16,256] encoder_out_lens:[32] hyps_pad_sos_eos:[32,10,64] hyps_lens_sos:[32,10] ctc_score:[32,10] \
    --obey-precision-constraints



  echo "Finish generate deploy.py"
  python /target/depoly.py


  https://code.aliyun.com/hushonion/wenet-TensorRT.git