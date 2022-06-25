import onnx
import onnx_graphsurgeon as gs
import numpy as np

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

@gs.Graph.register()
def replace_masked_softmax(self, inputs, outputs, name):

    for out in outputs:
        out.inputs.clear()

    # Insert the new node.
    return self.layer(op="AttnMaskedSoftmax",
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

cross_attn_nodes = [
        {"inps" : ["476",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1793", "decoder.decoders.0.src_attn.linear_q.bias",
                "1797", "decoder.decoders.0.src_attn.linear_k.bias",
                "1801", "decoder.decoders.0.src_attn.linear_v.bias",
                "1807", "decoder.decoders.0.src_attn.linear_out.bias",
                "decoder.decoders.0.norm2.weight",
                "decoder.decoders.0.norm2.bias",
                ],
        "outs" : ["575"]},
        {"inps" : ["693",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1825", "decoder.decoders.1.src_attn.linear_q.bias",
                "1829", "decoder.decoders.1.src_attn.linear_k.bias",
                "1833", "decoder.decoders.1.src_attn.linear_v.bias",
                "1839", "decoder.decoders.1.src_attn.linear_out.bias",
                "decoder.decoders.1.norm2.weight",
                "decoder.decoders.1.norm2.bias",
                ],
        "outs" : ["792"]},
        {"inps" : ["910",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1857", "decoder.decoders.2.src_attn.linear_q.bias",
                "1861", "decoder.decoders.2.src_attn.linear_k.bias",
                "1865", "decoder.decoders.2.src_attn.linear_v.bias",
                "1871", "decoder.decoders.2.src_attn.linear_out.bias",
                "decoder.decoders.2.norm2.weight",
                "decoder.decoders.2.norm2.bias",
                ],
        "outs" : ["1009"]},
        {"inps" : ["1127",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1889", "decoder.decoders.3.src_attn.linear_q.bias",
                "1893", "decoder.decoders.3.src_attn.linear_k.bias",
                "1897", "decoder.decoders.3.src_attn.linear_v.bias",
                "1903", "decoder.decoders.3.src_attn.linear_out.bias",
                "decoder.decoders.3.norm2.weight",
                "decoder.decoders.3.norm2.bias",
                ],
        "outs" : ["1226"]},
        {"inps" : ["1344",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1921", "decoder.decoders.4.src_attn.linear_q.bias",
                "1925", "decoder.decoders.4.src_attn.linear_k.bias",
                "1929", "decoder.decoders.4.src_attn.linear_v.bias",
                "1935", "decoder.decoders.4.src_attn.linear_out.bias",
                "decoder.decoders.4.norm2.weight",
                "decoder.decoders.4.norm2.bias",
                ],
        "outs" : ["1443"]},
        {"inps" : ["1561",       #   q
                "214",          #   enc_in
                "cross_attn_mask",      #   mask
                "1953", "decoder.decoders.5.src_attn.linear_q.bias",
                "1957", "decoder.decoders.5.src_attn.linear_k.bias",
                "1961", "decoder.decoders.5.src_attn.linear_v.bias",
                "1967", "decoder.decoders.5.src_attn.linear_out.bias",
                "decoder.decoders.5.norm2.weight",
                "decoder.decoders.5.norm2.bias",
                ],
        "outs" : ["1660"]},
    ]


def parent(node, num=0):
    if node.inputs[num].inputs[0].op in ["QuantizeLinear", "DequantizeLinear"]:
        return node.inputs[num].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0]
    else:
        return node.inputs[num].inputs[0]


def find_masked_softmax_nodes(graph):
    out_nodes = []
    for node in graph.nodes:
        if node.op == "Softmax":
            if node.inputs[0].inputs[0].op == "Where" and \
                    node.inputs[0].inputs[0].inputs[2].inputs[0].op == "Div" and \
                    node.outputs[0].outputs[0].op == "Where":

                div_node = node.inputs[0].inputs[0].inputs[2].inputs[0]
                qk_mm_node = parent(div_node, 0)
                q_mm_node = parent(parent(parent(parent(qk_mm_node, 0)), 0), 1)
                k_mm_node = parent(parent(parent(parent(qk_mm_node, 1)), 0), 1)
                attn_mask = "self_attn_mask"
                if q_mm_node.inputs[0].name != k_mm_node.inputs[0].name:
                    attn_mask = "cross_attn_mask"
                out_nodes += [{
                    "inps": [node.inputs[0].inputs[0].inputs[2].inputs[0].inputs[0].name,
                             attn_mask, ],
                    "outs": [node.outputs[0].outputs[0].outputs[0].name],
                }]

    return out_nodes

if __name__ == "__main__":
    input_mdl = './decoder_quant_dynamic.onnx'
    output_mdl = './decoder_replace.onnx'
    graph = gs.import_onnx(onnx.load(input_mdl))
    graph.inputs[2].shape = ["B",10, 64]

    self_attn_mask = gs.Variable(name="self_attn_mask", shape=["B_Attn", 63, 63], dtype=np.float32)
    cross_attn_mask = gs.Variable(name="cross_attn_mask", shape=["B_Attn", 63, "T"], dtype=np.float32)
    graph.inputs.extend([self_attn_mask])
    graph.inputs.extend([cross_attn_mask])
    tmap = graph.tensors()

    out_nodes = find_masked_softmax_nodes(graph)
    for i,itn in enumerate(out_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "masked_softmax_{}".format(i)
        graph.replace_masked_softmax(inputs, outputs, name)

    layer_norm_nodes = find_layer_norm_nodes(graph)
    for i,itn in enumerate(layer_norm_nodes):
        inputs = itn['inps']
        outputs = itn['outs']
        name = "layer_norm_{}".format(i)
        graph.replace_layer_norm(inputs, outputs, name)



    graph.cleanup().toposort()


    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output_mdl)





