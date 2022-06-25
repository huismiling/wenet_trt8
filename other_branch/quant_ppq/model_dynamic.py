import onnx
import onnx_graphsurgeon as gs
import numpy as np
import sys

flag = sys.argv[1]

ori = f'../../model/{flag}.onnx'


ori_model = onnx.load(ori)
gs_graph = gs.import_onnx(ori_model)

nameIn1 = [i.name for i in gs_graph.inputs]
nameOut1 = [i.name for i in gs_graph.outputs]

model_pb = ori_model.graph

dyDict = {'inpMess': {}, 'outMess': {}}
for item in model_pb.input:
    dyDict['inpMess'][item.name] = {}
    shape = item.type.tensor_type.shape.dim
    for i, j in enumerate(shape):
        if j.dim_param:
            dyDict['inpMess'][item.name][i] = j.dim_param
for item in model_pb.output:
    dyDict['outMess'][item.name] = {}
    shape = item.type.tensor_type.shape.dim
    for i, j in enumerate(shape):
        if j.dim_param:
            dyDict['outMess'][item.name][i] = j.dim_param



save = f'./{flag}_quant.onnx' if flag == 'decoder' else f'./{flag}_new.onnx'

model = onnx.load(save)

gs_graph = gs.import_onnx(model)

nameIn2 = [i.name for i in gs_graph.inputs]
nameOut2 = [i.name for i in gs_graph.outputs]

idxIn = [nameIn2.index(i) for i in nameIn1]
idxOut = [nameOut2.index(i) for i in nameOut1]


gs_graph.outputs = [gs_graph.outputs[idx] for idx in idxOut]
gs_graph.inputs = [gs_graph.inputs[idx] for idx in idxIn]

gs_graph.cleanup().toposort()

model = gs.export_onnx(gs_graph)
model_pb = model.graph

inpMess = model_pb.input
outMess = model_pb.output

if dyDict:
    for k, v in dyDict.items():
        for i in eval(k):
            for index, val in v[i.name].items():
                i.type.tensor_type.shape.dim[int(index)].dim_param = val

save = f'./{flag}_quant_dynamic.onnx'
onnx.save(model, save)