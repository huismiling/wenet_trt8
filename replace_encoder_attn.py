

import onnx
import onnx_graphsurgeon as gs

import numpy as np
from collections import OrderedDict



@gs.Graph.register()
def replace_attn(self, inputs, outputs, name, attrs):
    # Disconnect output nodes of all input tensors
    # for inp in inputs:
    #     inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="MultiHeadAttn", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    attrs=attrs,
                    )

@gs.Graph.register()
def replace_layer_norm(self, inputs, outputs, name):
    # Disconnect output nodes of all input tensors
    # for inp in inputs:
    #     inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="LayerNormPlugin", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    )


def find_node(graph, name):
    for node in graph.nodes:
        if node.name == name:
            return node

@gs.Graph.register()
def replace_div_2_mul(self, inputs, outputs, name):
    # Disconnect output nodes of all input tensors
    # for inp in inputs:
    #     inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="Mul", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    )


def find_masked_softmax_nodes(graph,speech_lengths_mask):
    out_nodes = []
    for node in graph.nodes:
        if node.op == "Softmax":
            if node.i().op == 'Where' and node.o().op == 'Where' and node.i().i(2).op == 'Div':
                out_nodes += [{
                    "inps": [node.i().i(2).inputs[0],
                             speech_lengths_mask, ],
                    "outs": [node.o().outputs[0]],
                }]
    
    return out_nodes

def find_layer_norm_nodes(graph):
    out_nodes = []
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):
            div_node = node.o().o(0).o().o().o().o()
            mul_node = div_node.o()
            add_node = div_node.o().o()
            out_nodes += [{
                "inps": [node.inputs[0],
                         mul_node.inputs[1],
                         add_node.inputs[1], ],
                "outs": [add_node.outputs[0]],
            }]


    return out_nodes


@gs.Graph.register()
def replace_masked_softmax(self, inputs, outputs, name):
    # Disconnect output nodes of all input tensors
    # for inp in inputs:
    #     inp.outputs.clear()

    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="AttnMaskedSoftmax", 
                    inputs=inputs, 
                    outputs=outputs,
                    name=name,
                    )


cross_attn_nodes = [
    ]

self_attn_nodes = [
    ]


div_2_mul_nodes =[
    "Div_156", "Div_313", "Div_470", "Div_627", "Div_784", 
    "Div_941", "Div_1098", "Div_1255", "Div_1412",  
    "Div_1569", "Div_1726", "Div_1883", "Div_1977", 
]

if __name__ == "__main__":
    import sys
    input_mdl = sys.argv[1]
    output_mdl = sys.argv[2]
    graph = gs.import_onnx(onnx.load(input_mdl))

#     graph.inputs = [graph.inputs[1], graph.inputs[0]]
    self_attn_mask = gs.Variable(name="speech_lengths_mask", shape=["B", "TM", "TM"], dtype=np.float32)
    graph.inputs.extend([self_attn_mask])
    # tmap = graph.tensors()
    # You can figure out the input and output tensors using Netron. In our case:
    # Inputs: [inp, MIN_VAL, MAX_VAL]
    # Outputs: [max_out]
#     for i,itn in enumerate(cross_attn_nodes):
#         inputs = [tmap[i] for i in itn["inps"]]
#         outputs = [tmap[i] for i in itn["outs"]]
#         name = "cross_attn_{}".format(i)
#         attrs = {"AttentionType":"cross"}
#         graph.replace_attn(inputs, outputs, name, attrs)

#     for i,itn in enumerate(self_attn_nodes):
#         inputs = [tmap[i] for i in itn["inps"]]
#         outputs = [tmap[i] for i in itn["outs"]]
#         name = "self_attn_{}".format(i)
#         attrs = {"AttentionType":"self"}
#         graph.replace_attn(inputs, outputs, name, attrs)

    layer_norm_nodes = find_layer_norm_nodes(graph)
    for i,itn in enumerate(layer_norm_nodes):
        inputs = itn['inps']
        outputs = itn['outs']
        name = "layer_norm_{}".format(i)
        graph.replace_layer_norm(inputs, outputs, name)

    # for itn, itd in enumerate(div_2_mul_nodes):
    #     div_node = find_node(graph, itd)
    #     print(div_node)
    #     div_node.op = "Mul"
    #     ci = gs.Constant("Div2Mul_{}".format(itn), np.array(0.125, dtype=np.float32))
    #     div_node.inputs[1] = ci

    out_nodes = find_masked_softmax_nodes(graph,self_attn_mask)
    for i,itn in enumerate(out_nodes):
        inputs = itn['inps']
        outputs = itn['outs']
        name = "masked_softmax_{}".format(i)
        graph.replace_masked_softmax(inputs, outputs, name)


    # # Remove the now-dangling subgraph.
    # graph.inputs[5].shape = ['B_Attn','UNK','UNK']
    # graph.inputs[6].shape = ['B_Attn','UNK','T']
    graph.cleanup().toposort()
#     graph.inputs[0].shape=[1, 16, 80]
#     graph.inputs[1].shape=[16, ]
    
    Relu_38 = Transpose_51 = Concat_59 = Reshape_60 = None
    for node in graph.nodes:
        if node.op == 'Relu' and node.name == 'Relu_38':
            Relu_38 = node
        if node.op == 'Transpose' and node.name == 'Transpose_51':
            Transpose_51 = node
        if node.op == 'Concat' and node.name == 'Concat_59':
            Concat_59 = node
        if node.op == 'Reshape' and node.name == 'Reshape_60':
            Reshape_60 = node

    Concat_59.outputs.clear()
    ShapeX_in = Transpose_51.outputs[0]
    ShapeX_out = gs.Variable(name='ShapeX_out',dtype=None,shape=None)
    ShapeX = gs.Node('Shape','ShapeX',inputs=[ShapeX_in],outputs=[ShapeX_out])
    graph.nodes.append(ShapeX)
    SliceX_out = gs.Variable(name='SliceX_out',dtype=None,shape=None)
    SliceX = gs.Node('Slice','SliceX',inputs=[ShapeX_out,gs.Constant(name='SliceX1',values=np.array([0])),
                                              gs.Constant(name='SliceX2',values=np.array([-2])),
                                              gs.Constant(name='SliceX3',values=np.array([0]))],
                     outputs=[SliceX_out])
    graph.nodes.append(SliceX)
    ConcatX_out = gs.Variable(name='ConcatX_out',dtype=None,shape=None)
    ConcatX = gs.Node('Concat','ConcatX',inputs=[SliceX_out,gs.Constant(name='ConcatX1',values=np.array([4864]))],
                     outputs=[ConcatX_out],attrs=OrderedDict(axis=0))
    graph.nodes.append(ConcatX)
    Reshape_60.inputs = [ShapeX_in,ConcatX_out]


    graph.cleanup().toposort()


    # That's it!
    onnx.save(gs.export_onnx(graph), output_mdl)





