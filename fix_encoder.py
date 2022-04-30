import numpy as np
import onnx
import onnx_graphsurgeon as gs
def wenet_encoder():
    encoder = onnx.load("/workspace/encoder.onnx")
    graph =  gs.import_onnx(encoder)
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
