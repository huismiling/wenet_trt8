# 英伟达TensorRT加速AI推理复赛—wenet

## 准备数据集和模型权重

1. 下载 `aishell` 测试集音频和标注文件，下载 torch 模型权重和导出过的 onnx，对数据集进行预处理，执行如下命令:

   ``` shell
   sh prepare_dataset.sh
   ```

2.  编译 TensorRT plugin 并对 onnx 进行优化，执行如下命令:

   ``` shell
   sh build.sh
   ```

3. 测试 pytorch 模型的推理精度。

   ``` shell
   cd wenet_repo/work_dir/shell
   ```

   进入到 shell 目录下修改 test-engine.sh, test-onnx.sh, test-pt.sh, cal_result.sh 文件中 `repoPath` 为当前 repo 的绝对路径，然后执行:

   ``` shell
   sh test-pt.sh
   sh test-onnx.sh
   sh test-engine.sh
   ```

   分别获取 pytorch onnx tensorrt 模型推理结果，保存到 `log/` 下。执行:

   ``` shell
   sh cal_result.sh
   ```

   分别获取  pytorch onnx tensorrt 推理结果的 `wer` 评分，保存在 `log/wer/` 下。查看文件最后两行即推理指标。