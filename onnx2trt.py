#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import numpy as np
from glob import glob
from datetime import datetime as dt
# import torch as t
# #import torchvision as tv               # 使用 pyTorch 默认的 MNIST 数据（含下载）
# from torch.utils import data
# import torch.nn.functional as F
# from torch.autograd import Variable
from cuda import cudart
import tensorrt as trt
# from trt_qant import calibrator
import ctypes

cacheFile = "./int8.cache"
calibrationCount = 10
np.set_printoptions(precision=4, linewidth=200, suppress=True)
cudart.cudaDeviceSynchronize()


planFilePath   = "./"
soFileList = ["./libmhalugin.so"]

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

# calibration.npz files
# 'speech-16', 'speech-64', 'speech-256', 
# 'speech_lengths-16', 'speech_lengths-64', 'speech_lengths-256', 
# 'encoder_out-16', 'encoder_out-64', 'encoder_out-256', 
# 'encoder_out_lens-16', 'encoder_out_lens-64', 'encoder_out_lens-256', 
# 'hyps_pad_sos_eos-16', 'hyps_pad_sos_eos-64', 'hyps_pad_sos_eos-256', 
# 'hyps_lens_sos-16', 'hyps_lens_sos-64', 'hyps_lens_sos-256', 
# 'ctc_score-16', 'ctc_score-64', 'ctc_score-256'

ckey = sys.argv[1]  # encoder or decoder
assert ckey in ["encoder", "decoder"]
onnxFile = sys.argv[2]
trtFile = sys.argv[3]
calibData = np.load("data/calibration.npz")

if ckey == "encoder":
    npDataList = [
        {
            "speech": calibData['speech-16'],
            "speech_lengths": calibData['speech_lengths-16']
        },
        # {
        #     "speech": calibData['speech-64'],
        #     "speech_lengths": calibData['speech_lengths-64']
        # },
        # {
        #     "speech": calibData['speech-256'],
        #     "speech_lengths": calibData['speech_lengths-256']
        # },
    ]
    inputShapes = {"speech": (1, 16, 80), "speech_lengths": (1,)}

# TensorRT 中加载 .onnx 创建 engine ----------------------------------------------
logger = trt.Logger(trt.Logger.INFO)
logger.min_severity = trt.Logger.Severity.VERBOSE
if 1:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    if 1 or ckey == "encoder":
        config.flags = 1 << int(trt.BuilderFlag.INT8)

    #     config.int8_calibrator = calibrator.MyCalibrator(npDataList, calibrationCount, inputShapes, cacheFile)
    # config.max_workspace_size = 1 << 50
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")

    if ckey == "decoder":
        profile.set_shape("encoder_out", (1, 40, 256), (4, 165, 256), (16, 370, 256))
        profile.set_shape("encoder_out_lens", (1,), (4,), (16,))
        profile.set_shape("hyps_pad_sos_eos", (1, 10, 64), (4, 10, 64), (16, 10, 64))
        profile.set_shape("hyps_lens_sos", (1, 10), (4, 10), (16, 10))
        profile.set_shape("ctc_score", (1, 10), (4, 10), (16, 10))
        profile.set_shape("self_attn_mask", (10, 63, 63), (40, 63, 63), (160, 63, 63))
        profile.set_shape("cross_attn_mask", (10, 63, 40), (40, 63, 165), (160, 63, 370))
    elif ckey == "encoder":
        profile.set_shape("speech", (1, 1, 80), (4, 750, 80), (16, 1500, 80))
        profile.set_shape("speech_lengths", (1,), (4,), (16,))
        profile.set_shape("speech_lengths_mask", (1, 40, 40), (4, 220, 220), (16, 400, 400))
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

