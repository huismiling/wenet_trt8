import onnxruntime as rt
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationMethod

from onnx_qant import onnxDataReader

import numpy as np
import sys

ckey = sys.argv[1]      # encoder or decoder
assert ckey in ["encoder", "decoder"]
mdl_path = sys.argv[2]
model_quant = sys.argv[3]
sess_options = rt.SessionOptions()

# Set graph optimization level
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

# To enable model serialization after graph optimization set this
# sess_options.optimized_model_filepath = opt_path

session = rt.InferenceSession(mdl_path, sess_options, providers=['CPUExecutionProvider'])




# quantized_model = quantize_dynamic(opt_path, model_quant)

# calibration.npz files
# 'speech-16', 'speech-64', 'speech-256', 
# 'speech_lengths-16', 'speech_lengths-64', 'speech_lengths-256', 
# 'encoder_out-16', 'encoder_out-64', 'encoder_out-256', 
# 'encoder_out_lens-16', 'encoder_out_lens-64', 'encoder_out_lens-256', 
# 'hyps_pad_sos_eos-16', 'hyps_pad_sos_eos-64', 'hyps_pad_sos_eos-256', 
# 'hyps_lens_sos-16', 'hyps_lens_sos-64', 'hyps_lens_sos-256', 
# 'ctc_score-16', 'ctc_score-64', 'ctc_score-256'

calibData = np.load("./data/calibration.npz")

if ckey == "encoder":
    NpData = [
        {'speech':calibData['speech-16'], 'speech_lengths':calibData['speech_lengths-16']},
        {'speech':calibData['speech-64'], 'speech_lengths':calibData['speech_lengths-64']},
        {'speech':calibData['speech-256'], 'speech_lengths':calibData['speech_lengths-256']},
        ]

    with open("encoder_quant_nodes.txt") as f:
        quant_nodes = [it.strip() for it in f.readlines()]
    with open("encoder_quant_exclude_nodes.txt") as f:
        exclue_nodes = [it.strip() for it in f.readlines()]
elif ckey == "decoder":
    NpData = [
        # {
        #     'encoder_out':calibData['encoder_out-16'], 
        #     'encoder_out_lens':calibData['encoder_out_lens-16'],
        #     'hyps_pad_sos_eos':calibData['hyps_pad_sos_eos-16'].astype(np.int64),
        #     'hyps_lens_sos':calibData['hyps_lens_sos-16'],
        #     'ctc_score':calibData['ctc_score-16'],
        # },
        {
            'encoder_out':calibData['encoder_out-64'], 
            'encoder_out_lens':calibData['encoder_out_lens-64'],
            'hyps_pad_sos_eos':calibData['hyps_pad_sos_eos-64'].astype(np.int64),
            'hyps_lens_sos':calibData['hyps_lens_sos-64'],
            'ctc_score':calibData['ctc_score-64'],
        },
        # {
        #     'encoder_out':calibData['encoder_out-256'], 
        #     'encoder_out_lens':calibData['encoder_out_lens-256'],
        #     'hyps_pad_sos_eos':calibData['hyps_pad_sos_eos-256'].astype(np.int64),
        #     'hyps_lens_sos':calibData['hyps_lens_sos-256'],
        #     'ctc_score':calibData['ctc_score-256'],
        # },
    ]
    quant_nodes = None
    exclue_nodes = None

    with open("decoder_quant_nodes.txt") as f:
        quant_nodes = [it.strip() for it in f.readlines()]
    with open("decoder_quant_exclude_nodes.txt") as f:
        exclue_nodes = [it.strip() for it in f.readlines()]
print('*'*30)
print(len(exclue_nodes))
print('*'*30)
calibrator = onnxDataReader(NpData, batch_size=1, run_times=200)
quantize_static(mdl_path, model_quant, calibrator, 
            nodes_to_quantize=quant_nodes, 
            nodes_to_exclude=None, # exclue_nodes, 
            # per_channel = True,
            extra_options={
                    "ActivationSymmetric": True,
                    "WeightSymmetric": True,
                    "AddQDQPairToWeight": True,
                },
            # calibrate_method=CalibrationMethod.Percentile,
            )
