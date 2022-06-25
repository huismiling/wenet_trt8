import numpy as np
import onnx
import onnx_graphsurgeon as gs
from collections import OrderedDict


path = "../../model/encoder.onnx"

encoder = onnx.load(path)
graph = gs.import_onnx(encoder)

Not_30 = Relu_38 = Transpose_51 = Reshape_60 = Slice_74 = Slice_79 = Slice_84 = None
for node in graph.nodes:
    if node.op == 'Not' and node.name == 'Not_30':
        Not_30 = node
    if node.op == 'Relu' and node.name == 'Relu_38':
        Relu_38 = node
    if node.op == 'Transpose' and node.name == 'Transpose_51':
        Transpose_51 = node
    if node.op == 'Reshape' and node.name == 'Reshape_60':
        Reshape_60 = node
    if node.op == 'Slice' and node.name == "Slice_74":
        Slice_74 = node
    if node.op == 'Slice' and node.name == "Slice_79":
        Slice_79 = node
    if node.op == 'Slice' and node.name == "Slice_84":
        Slice_84 = node

Slice_79.inputs[2] = gs.Constant(name=Slice_79.inputs[2].name, values=np.array([-6], dtype=np.int64))
Slice_79.inputs[4] = gs.Constant(name=Slice_79.inputs[4].name, values=np.array([4], dtype=np.int64))

Cast0_input = Not_30.outputs[0]
Cast0_output = gs.Variable(name="Cast0_output", dtype=np.dtype(np.int32), shape=None)

Cast0 = gs.Node(name='Add_Cast0', op='Cast',
                inputs=[Cast0_input],
                outputs=[Cast0_output],
                attrs=OrderedDict(to=6))
graph.nodes.append(Cast0)

Slice_79.inputs[0] = Cast0_output

Cast1_input = Slice_79.outputs[0]
Cast1_output = Slice_84.outputs[0]
Slice_84.outputs.clear()
Cast1 = gs.Node(name='Add_Cast1', op='Cast',
                inputs=[Cast1_input],
                outputs=[Cast1_output],
                attrs=OrderedDict(to=9))
graph.nodes.append(Cast1)

table5000x256 = np.load('../../tensor')[0]
t4Tensor = Slice_74.inputs[2]
zero = gs.Constant(name='Constant-0', values=np.array([0]))
one = gs.Constant(name='Constant-1', values=np.array([1]))
three = gs.Constant(name='Constant-3', values=np.array([3]))
j = 0
for i in range(1, 24, 2):
    trashNode = Slice_74.o(i).o().o()
    factor256x256 = Slice_74.o(i).inputs[1].values
    newTable = table5000x256 @ factor256x256
    newTable = newTable.transpose().reshape(1, 4, 64, 5000)
    constantData = gs.Constant(f'Data-{j}', np.ascontiguousarray(newTable))
    sliceV = gs.Variable(f'sliceData-{j}', np.dtype(np.float32), [1, 4, 64, 't4'])
    sliceN = gs.Node('Slice', f'SliceN-{j}',
                     inputs=[constantData, zero, t4Tensor, three, one],
                     outputs=[sliceV])
    j += 1
    graph.nodes.append(sliceN)
    Slice_74.o(i).o().o().o().inputs[1] = sliceV
    trashNode.outputs.clear()

ShapeX_in = Transpose_51.outputs[0]
ShapeX_out = gs.Variable(name='ShapeX_out', dtype=None, shape=None)
ShapeX = gs.Node('Shape', 'ShapeX', inputs=[ShapeX_in], outputs=[ShapeX_out])
graph.nodes.append(ShapeX)
SliceX_out = gs.Variable(name='SliceX_out', dtype=None, shape=None)
SliceX = gs.Node('Slice', 'SliceX', inputs=[ShapeX_out, gs.Constant(name='SliceX1', values=np.array([0])),
                                            gs.Constant(name='SliceX2', values=np.array([2])),
                                            gs.Constant(name='SliceX3', values=np.array([0]))],
                 outputs=[SliceX_out])
graph.nodes.append(SliceX)
ConcatX_out = gs.Variable(name='ConcatX_out', dtype=None, shape=None)
ConcatX = gs.Node('Concat', 'ConcatX', inputs=[SliceX_out, gs.Constant(name='ConcatX1', values=np.array([-1]))],
                  outputs=[ConcatX_out], attrs=OrderedDict(axis=0))
graph.nodes.append(ConcatX)
Reshape_60.inputs = [ShapeX_in, ConcatX_out]

graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "./encoder_new.onnx")





