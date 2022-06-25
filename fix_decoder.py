import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np
import onnx_graphsurgeon as gs

def get_quant_nodes(graph):
    quant_nodes = []
    exclude_nodes = [] # ["MatMul_178", "MatMul_141", "MatMul_119", "MatMul_125", 
                       #  "MatMul_131", "Transpose_173", "Reshape_177"]
    for node in graph.nodes:
        if node.op in ["Conv", "MatMul"]:
            quant_nodes.append(node.name)
        # if node.op == "MatMul" and \
        #     isinstance(node.inputs[1], gs.Constant):
        #     quant_nodes.append(node.name)

    # for node in graph.nodes:
    #     if node.op in ["Softmax", ]:
    #         print("decoder_quant_exclude_nodes: ", node.name)
    #         exclude_nodes.append(node.name)
    #     if node.op == "Add" and \
    #         "norm" in node.inputs[1].name:
    #         print("decoder_quant_exclude_nodes: ", node.name, " ", node.inputs[1].name)
    #         exclude_nodes.append(node.name)
    #     if node.op == "Mul" and \
    #         "norm" in node.inputs[1].name:
    #         print("encoder_quant_exclude_nodes: ", node.name, " ", node.inputs[1].name)
    #         exclude_nodes.append(node.name)

    with open("decoder_quant_nodes.txt", "w+") as f:
        f.write('\n'.join(quant_nodes))
    with open("decoder_quant_exclude_nodes.txt", "w+") as f:
        f.write('\n'.join(exclude_nodes))

mdl = onnx.load("model/decoder.onnx")
graph =  gs.import_onnx(mdl)
get_quant_nodes(graph)
graph.inputs[2].shape[2] = 64
print(graph.inputs[2])

onnx.save(gs.export_onnx(graph), "decoder_new.onnx")



