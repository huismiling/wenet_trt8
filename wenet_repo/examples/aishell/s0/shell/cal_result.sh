#!/bin/bash
set -x
cd ..
export PYTHONPATH=../../../:$PYTHONPATH

mode=${1:-attention_rescoring}
echo '**************************************************'
echo "mode is $mode"
echo '**************************************************'
#      # /home/ubuntu/study/wenet/work_dir/result/$mode/text \
#      > /home/ubuntu/study/wenet/work_dir/result/$mode/wer
python tools/compute-wer.py \
     --char=1 \
     --v=1 \
      /home/ubuntu/study/wenet/work_dir/data/test/text \
      /home/ubuntu/study/wenet/work_dir/onnx_export/onnx_result.txt \
      > /home/ubuntu/study/wenet/work_dir/onnx_export/onnx_result.wer