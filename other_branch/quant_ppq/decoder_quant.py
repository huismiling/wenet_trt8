import torch
import onnx
import random
from ppq import *
from ppq.api import *
from ppq.quantization.quantizer import TensorRTQuantizer
from data import MyDataSet
from torch.utils.data import DataLoader


DEVICE = 'cuda'
PLATFORM = TargetPlatform.TRT_INT8

save = './decoder_quant.onnx'
MODEL_PATH = path = "../../model/decoder.onnx"

dataset = MyDataSet(mode='decoder')

SAMPLES = DataLoader(dataset,
                    batch_size=None,
                    shuffle=True,
                    num_workers=0)

inp = None
for _ in range(random.randint(1,100)):
    inp =  next(iter(SAMPLES))
print(inp)

with ENABLE_CUDA_KERNEL():
    QS = QuantizationSettingFactory.trt_setting()
    ir = load_onnx_graph(onnx_import_file=MODEL_PATH)
    for op in ir.operations.values():
        if op.type in {'Unsqueeze', 'Squeeze', 'ReduceSum'}:
            axes = op.inputs[1].value
            ir.remove_variable(removing_var=op.inputs[1])
            op.attributes['axes'] = axes.tolist()

    qir = quantize_native_model(
        model=ir, calib_dataloader=SAMPLES, calib_steps=300,
        input_shape=None, inputs=inp,
        platform=TargetPlatform.TRT_INT8, setting=QS)

    graphwise_error_analyse(
        graph=qir, running_device='cuda',
        dataloader=SAMPLES)

    export_ppq_graph(graph=qir, platform=TargetPlatform.TRT_INT8,
                     graph_save_to=save)
