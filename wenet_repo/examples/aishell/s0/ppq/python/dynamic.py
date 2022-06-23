import onnx
import onnx_graphsurgeon as gs
import numpy as np

flag = 'encoder'

ori = f'../orin_onnx/{flag}.onnx'
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



save = f'../quant_onnx/{flag}.onnx'

model = onnx.load(save)

save = f'../quant_onnx/{flag}_quant.onnx'

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

onnx.save(model, save)

'''
onnxsim input.onnx output.onnx \
--input-shape "aaa:1,70,256" "bbb:1,200,200,3" "ccc:1,5,600"
'''

'''
trtexec --onnx=encoder_quant.onnx --saveEngine=./encoder.plan \
    --minShapes=speech:1x16x80,speech_lengths:1 \
    --optShapes=speech:4x64x80,speech_lengths:4 \
    --maxShapes=speech:16x256x80,speech_lengths:16 \
    --workspace=23028 --verbose \
    2>&1 | tee log_encoder.txt
'''