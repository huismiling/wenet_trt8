import onnx
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_layer_norm(self, inputs, outputs, name):


    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="LayerNormPlugin",
                      inputs=inputs,
                      outputs=outputs,
                      name=name,
                      )


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


if __name__ == "__main__":
    input_mdl = './encoder_quant_dynamic.onnx'
    output_mdl = './encoder_replace.onnx'
    graph = gs.import_onnx(onnx.load(input_mdl))

    layer_norm_nodes = find_layer_norm_nodes(graph)
    for i,itn in enumerate(layer_norm_nodes):
        inputs = itn['inps']
        outputs = itn['outs']
        name = "layer_norm_{}".format(i)
        graph.replace_layer_norm(inputs, outputs, name)

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_mdl)





