import numpy as np
import onnx
import onnx_graphsurgeon as gs

def get_quant_nodes(graph):
    quant_nodes = []
    exclude_nodes = [] # ["MatMul_178", "MatMul_141", "MatMul_119", "MatMul_125", 
                       #  "MatMul_131", "Transpose_173", "Reshape_177"]
    for node in graph.nodes:
        if node.op in ["Conv"]:  
            if node.attrs['group']==1:
                quant_nodes.append(node.name)
        if node.op == "MatMul" and \
            isinstance(node.inputs[1], gs.Constant):
            quant_nodes.append(node.name)

    for node in graph.nodes:
        if node.op in ["Softmax", ]:
            print("encoder_quant_exclude_nodes: ", node.name)
            exclude_nodes.append(node.name)
        if node.op == "Add" and \
            "norm" in node.inputs[1].name:
            print("encoder_quant_exclude_nodes: ", node.name, " ", node.inputs[1].name)
            exclude_nodes.append(node.name)
        if node.op == "Mul" and \
            "norm" in node.inputs[1].name:
            print("encoder_quant_exclude_nodes: ", node.name, " ", node.inputs[1].name)
            exclude_nodes.append(node.name)

    with open("encoder_quant_nodes.txt", "w+") as f:
        f.write('\n'.join(quant_nodes))
    with open("encoder_quant_exclude_nodes.txt", "w+") as f:
        f.write('\n'.join(exclude_nodes))

def wenet_encoder():
    encoder = onnx.load("model/encoder.onnx")
    graph =  gs.import_onnx(encoder)
    get_quant_nodes(graph)
    Unsqueeze_29 = None
    for node in graph.nodes:
        if node.op == 'Unsqueeze' and node.name == "Unsqueeze_29":
            Unsqueeze_29 = node
        if node.op == 'Not' and node.name == 'Not_30':
            Not_30 = node
        if node.op == 'Slice' and node.name == "Slice_79":
            Slice_79 = node
        if node.op == 'Slice' and node.name == "Slice_84":
            Slice_84 = node
    start_node = Unsqueeze_29.outputs[0]
    Unsqueeze_29_Cast_output = gs.Variable(name="Unsqueeze_29_Cast_output", dtype=None, shape=None)
    attrs_dict = {}
    attrs_dict['to'] = 6
    newNode = gs.Node(name="Slice_79_Cast", op="Cast", inputs=[start_node],
                      outputs=[Unsqueeze_29_Cast_output], attrs=attrs_dict)
    graph.nodes.append(newNode)  # 记得把新节点加入计算图中

    Slice_79.inputs[0] = Unsqueeze_29_Cast_output
    Slice_84_outputs = Not_30.outputs[0]
    end_node = Slice_84.outputs[0]
    Not_30.outputs[0] = end_node
    Slice_84.outputs[0] = Slice_84_outputs
    Not_30.inputs[0] = Slice_84.outputs[0]

    Slice_84_Cast_output = gs.Variable(name="Slice_84_Cast_output", dtype=None, shape=None)
    attrs_dict = {}
    attrs_dict['to'] = 9
    newNode = gs.Node(name="Slice_84_Cast", op="Cast", inputs=[Slice_84_outputs ],
                      outputs=[Slice_84_Cast_output], attrs=attrs_dict)
    graph.nodes.append(newNode)  # 记得把新节点加入计算图中
    Not_30.inputs[0] = Slice_84_Cast_output
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), "encoder_new.onnx")
    pass
if __name__ == '__main__':
    wenet_encoder()
