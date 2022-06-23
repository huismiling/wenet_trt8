import random
from dataloader import MyDataSet
from torch.utils.data import DataLoader
from ppq.core.config import PPQ_CONFIG
from ppq.api import load_onnx_graph
from ppq.core import TargetPlatform
from ppq.quantization.quantizer import TensorRTQuantizer
from ppq import QuantizationSettingFactory
from ppq.api.interface import QUANTIZER_COLLECTION,export_ppq_graph,quantize_native_model,register_network_quantizer,ENABLE_CUDA_KERNEL
# from ppq.executor import TorchExecutor
from ppq.quantization.analyse import layerwise_error_analyse, graphwise_error_analyse


PPQ_CONFIG.USING_CUDA_KERNEL = True
target_platform = TargetPlatform.TRT_INT8

flag = 'encoder'

dataset = MyDataSet(mode=flag)
SAMPLES = DataLoader(dataset,
                    batch_size=None,
                    shuffle=True,
                    num_workers=0)

inp = None
for _ in range(random.randint(1,100)):
    inp =  next(iter(SAMPLES))
print(inp)

model_path = '../tmp_onnx/encoder-sim.onnx'


register_network_quantizer(
    quantizer=TensorRTQuantizer,
    platform=TargetPlatform.TRT_INT8)
QUANTIZER_COLLECTION[TargetPlatform.TRT_INT8] = TensorRTQuantizer




with ENABLE_CUDA_KERNEL():
    QS = QuantizationSettingFactory.trt_setting()
    ppq_graph_ir = load_onnx_graph(model_path)


    for op in ppq_graph_ir.operations.values():
        if op.type in {'Unsqueeze', 'Squeeze', 'ReduceSum'}:
            axes = op.inputs[1].value
            ppq_graph_ir.remove_variable(removing_var=op.inputs[1])
            op.attributes['axes'] = axes.tolist()



    # quantizer = QUANTIZER_COLLECTION[target_platform](graph=ppq_graph_ir)

    # executor = TorchExecutor(ppq_graph_ir, device='cuda') # for cuda execution
    # quantizer.quantize(
    #         inputs=inp,                         # some random input tensor, should be list or dict for multiple inputs
    #         calib_dataloader=SAMPLES,                # calibration dataloader
    #         executor=executor,                          # executor in charge of everywhere graph execution is needed
    #         setting=QS,                            # quantization setting
    #         calib_steps=300                  # number of batched data needed in calibration, 8~512
    # )


    qir = quantize_native_model(
        model=ppq_graph_ir, calib_dataloader=SAMPLES, calib_steps=300,
        input_shape=None, inputs=inp,
        platform=TargetPlatform.TRT_INT8, setting=QS)

    graphwise_error_analyse(
        graph=qir, # ppq ir graph
        running_device='cuda', # cpu or cuda
        method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
        steps=2, # how many batches of data will be used for error analysis
        dataloader=SAMPLES
    )

    layerwise_error_analyse(
        graph=qir,
        running_device='cuda',
        method='snr',  # the metric is signal noise ratio by default, adjust it to 'cosine' if that's desired
        steps=2,
        dataloader=SAMPLES
    )

    export_ppq_graph(graph=qir, platform=TargetPlatform.TRT_INT8,
                         graph_save_to=f'../quant_onnx/{flag}.onnx',config_save_to=f'../log/{flag}.json')



