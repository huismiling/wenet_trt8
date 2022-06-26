#!/bin/bash
set -x
cd ..
export PYTHONPATH=../../:$PYTHONPATH
mode=${1:-attention_rescoring}

# # pytorch result test
# python tools/compute-wer.py \
#      --char=1 \
#      --v=1 \
#       $repoPath/datasets/text \
#       $repoPath/log/torch_result.txt \
#       2>&1 | tee $repoPath/log/wer/pytorch.txt

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
      $repoPath/log/engine1_result.txt \
      2>&1 | tee $repoPath/log/wer/engine1.txt

python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      $repoPath/datasets/text \
      $repoPath/log/engine2_result.txt \
      2>&1 | tee $repoPath/log/wer/engine2.txt

python wenet/bin/compute-speed.py \
      $repoPath/log/npys \
      onnx \
      $mode

python wenet/bin/compute-speed.py \
      $repoPath/log/npys \
      engine1 \
      $mode

python wenet/bin/compute-speed.py \
      $repoPath/log/npys \
      engine2 \
      $mode