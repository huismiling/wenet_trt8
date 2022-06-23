#!/bin/bash
set -x
cd ..
export PYTHONPATH=../../:$PYTHONPATH
repoPath=/workspace/wenet_trt8

# pytorch result test
python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      $repoPath/datasets/text \
      $repoPath/log/torch_result.txt \
      2>&1 | tee $repoPath/log/wer/pytorch.txt

# onnx result test
python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      $repoPath/datasets/text \
      $repoPath/log/onnx_result.txt \
      2>&1 | tee $repoPath/log/wer/onnx.txt

# tensorrt result test
python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      $repoPath/datasets/text \
      $repoPath/log/tensorrt_result.txt \
      2>&1 | tee $repoPath/log/wer/tensorrt.txt