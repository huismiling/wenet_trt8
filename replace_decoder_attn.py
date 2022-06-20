

import onnx
import onnx_graphsurgeon as gs

import numpy as np
import sys



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
                    "inps":[node.inputs[0].inputs[0].inputs[2].inputs[0].inputs[0].name,
                            attn_mask,],
                    "outs" : [node.outputs[0].outputs[0].outputs[0].name],
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

self_attn_nodes = [
        {"inps" : ["377",       #   q
                "377",          #   enc_in
                "self_attn_mask",      #   mask
                "1778", "decoder.decoders.0.self_attn.linear_q.bias",
                "1782", "decoder.decoders.0.self_attn.linear_k.bias", 
                "1786", "decoder.decoders.0.self_attn.linear_v.bias",
                "1792", "decoder.decoders.0.self_attn.linear_out.bias",
                "decoder.decoders.0.norm1.weight",
                "decoder.decoders.0.norm1.bias",
                ],
        "outs" : ["476"]},
        {"inps" : ["594",       #   q
                "594",          #   enc_in
                "self_attn_mask",      #   mask
                "1810", "decoder.decoders.1.self_attn.linear_q.bias",
                "1814", "decoder.decoders.1.self_attn.linear_k.bias", 
                "1818", "decoder.decoders.1.self_attn.linear_v.bias",
                "1824", "decoder.decoders.1.self_attn.linear_out.bias",
                "decoder.decoders.1.norm1.weight",
                "decoder.decoders.1.norm1.bias",
                ],
        "outs" : ["693"]},
        {"inps" : ["811",       #   q
                "811",          #   enc_in
                "self_attn_mask",      #   mask
                "1842", "decoder.decoders.2.self_attn.linear_q.bias",
                "1846", "decoder.decoders.2.self_attn.linear_k.bias", 
                "1850", "decoder.decoders.2.self_attn.linear_v.bias",
                "1856", "decoder.decoders.2.self_attn.linear_out.bias",
                "decoder.decoders.2.norm1.weight",
                "decoder.decoders.2.norm1.bias",
                ],
        "outs" : ["910"]},
        {"inps" : ["1028",       #   q
                "1028",          #   enc_in
                "self_attn_mask",      #   mask
                "1874", "decoder.decoders.3.self_attn.linear_q.bias",
                "1878", "decoder.decoders.3.self_attn.linear_k.bias", 
                "1882", "decoder.decoders.3.self_attn.linear_v.bias",
                "1888", "decoder.decoders.3.self_attn.linear_out.bias",
                "decoder.decoders.3.norm1.weight",
                "decoder.decoders.3.norm1.bias",
                ],
        "outs" : ["1127"]},
        {"inps" : ["1245",       #   q
                "1245",          #   enc_in
                "self_attn_mask",      #   mask
                "1906", "decoder.decoders.4.self_attn.linear_q.bias",
                "1910", "decoder.decoders.4.self_attn.linear_k.bias", 
                "1914", "decoder.decoders.4.self_attn.linear_v.bias",
                "1920", "decoder.decoders.4.self_attn.linear_out.bias",
                "decoder.decoders.4.norm1.weight",
                "decoder.decoders.4.norm1.bias",
                ],
        "outs" : ["1344"]},
        {"inps" : ["1462",       #   q
                "1462",          #   enc_in
                "self_attn_mask",      #   mask
                "1938", "decoder.decoders.5.self_attn.linear_q.bias",
                "1942", "decoder.decoders.5.self_attn.linear_k.bias", 
                "1946", "decoder.decoders.5.self_attn.linear_v.bias",
                "1952", "decoder.decoders.5.self_attn.linear_out.bias",
                "decoder.decoders.5.norm1.weight",
                "decoder.decoders.5.norm1.bias",
                ],
        "outs" : ["1561"]},
    ]

layer_norm_nodes = [
        {"inps" : ["575",       #   q
                "decoder.decoders.0.norm3.weight",
                "decoder.decoders.0.norm3.bias",
                ],
        "outs" : ["586"]},
        {"inps" : ["792",       #   q
                "decoder.decoders.1.norm3.weight",
                "decoder.decoders.1.norm3.bias",
                ],
        "outs" : ["803"]},
        {"inps" : ["1009",       #   q
                "decoder.decoders.2.norm3.weight",
                "decoder.decoders.2.norm3.bias",
                ],
        "outs" : ["1020"]},
        {"inps" : ["1226",       #   q
                "decoder.decoders.3.norm3.weight",
                "decoder.decoders.3.norm3.bias",
                ],
        "outs" : ["1237"]},
        {"inps" : ["1443",       #   q
                "decoder.decoders.4.norm3.weight",
                "decoder.decoders.4.norm3.bias",
                ],
        "outs" : ["1454"]},
        {"inps" : ["1660",       #   q
                "decoder.decoders.5.norm3.weight",
                "decoder.decoders.5.norm3.bias",
                ],
        "outs" : ["1671"]},
        {"inps" : ["1679",       #   q
                "decoder.after_norm.weight",
                "decoder.after_norm.bias",
                ],
        "outs" : ["1690"]},
]

if __name__ == "__main__":
    input_mdl = sys.argv[1]
    output_mdl = sys.argv[2]
    graph = gs.import_onnx(onnx.load(input_mdl))

    self_attn_mask = gs.Variable(name="self_attn_mask", shape=["B_Attn", 63, 63], dtype=np.float32)
    cross_attn_mask = gs.Variable(name="cross_attn_mask", shape=["B_Attn", 63, "T"], dtype=np.float32)
    graph.inputs.extend([self_attn_mask])
    graph.inputs.extend([cross_attn_mask])
    tmap = graph.tensors()
    # You can figure out the input and output tensors using Netron. In our case:
    # Inputs: [inp, MIN_VAL, MAX_VAL]
    # Outputs: [max_out]
    out_nodes = find_masked_softmax_nodes(graph)
    for i,itn in enumerate(out_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "masked_softmax_{}".format(i)
        graph.replace_masked_softmax(inputs, outputs, name)

    # for i,itn in enumerate(cross_attn_nodes):
    #     inputs = [tmap[i] for i in itn["inps"]]
    #     outputs = [tmap[i] for i in itn["outs"]]
    #     name = "cross_attn_{}".format(i)
    #     attrs = {"AttentionType":"cross"}
    #     graph.replace_attn(inputs, outputs, name, attrs)

    # for i,itn in enumerate(self_attn_nodes):
    #     inputs = [tmap[i] for i in itn["inps"]]
    #     outputs = [tmap[i] for i in itn["outs"]]
    #     name = "self_attn_{}".format(i)
    #     attrs = {"AttentionType":"self"}
    #     graph.replace_attn(inputs, outputs, name, attrs)

    for i,itn in enumerate(layer_norm_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "layer_norm_{}".format(i)
        graph.replace_layer_norm(inputs, outputs, name)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

    # That's it!
    onnx.save(gs.export_onnx(graph), output_mdl)





