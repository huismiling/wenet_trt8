

import onnx
import onnx_graphsurgeon as gs

import numpy as np



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


def find_masked_softmax_nodes(graph):
    out_nodes = []
    for node in graph.nodes:
        if node.op == "Softmax":
            if node.inputs[0].inputs[0].op == "Where" and \
                node.inputs[0].inputs[0].inputs[2].inputs[0].op == "Div" and \
                    node.outputs[0].outputs[0].op == "Where":
                out_nodes += [{
                    "inps":[node.inputs[0].inputs[0].inputs[2].inputs[0].inputs[0].name,
                            "speech_lengths_mask",],
                    "outs" : [node.outputs[0].outputs[0].outputs[0].name],
                }]
    
    return out_nodes

def find_layer_norm_nodes(graph):
    out_nodes = []
    for node in graph.nodes:
        if node.op == "ReduceMean":
            if node.outputs[0].outputs[0].op == "Sub" and \
                node.outputs[0].outputs[0].inputs[0].name == node.inputs[0].name:
                sub_node = node.outputs[0].outputs[0]
                pow_node = sub_node.outputs[0].outputs[0]
                div_node = sub_node.outputs[0].outputs[1]
                mean_node = pow_node.outputs[0].outputs[0]
                add_node = mean_node.outputs[0].outputs[0]
                sqrt_node = add_node.outputs[0].outputs[0]
                if div_node.op == "Div" and \
                    pow_node.op == "Pow" and \
                    mean_node.op == "ReduceMean" and \
                    add_node.op == "Add" and \
                    sqrt_node.op == "Sqrt" and \
                    sqrt_node.outputs[0].outputs[0].name == div_node.name:
                    mul_node = div_node.outputs[0].outputs[0]
                    add0_node = mul_node.outputs[0].outputs[0]
                    if mul_node.op == "Mul" and add0_node.op == "Add":
                        out_nodes += [{
                            "inps":[node.inputs[0].name,
                                    mul_node.inputs[1].name,
                                    add0_node.inputs[1].name,],
                            "outs" : [add0_node.outputs[0].name],
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

layer_norm_nodes = [
    {"inps" : ["587",       #   q
            "encoder.encoders.0.norm_ff_macaron.weight",
            "encoder.encoders.0.norm_ff_macaron.bias",
            ],
    "outs" : ["624"]},
    {"inps" : ["635",       #   q
            "encoder.encoders.0.norm_mha.weight",
            "encoder.encoders.0.norm_mha.bias",
            ],
    "outs" : ["646"]},
    {"inps" : ["758",       #   q
            "encoder.encoders.0.norm_conv.weight",
            "encoder.encoders.0.norm_conv.bias",
            ],
    "outs" : ["769"]},
    {"inps" : ["790",       #   q
            "encoder.encoders.0.norm_ff.weight",
            "encoder.encoders.0.norm_ff.bias",
            ],
    "outs" : ["801"]},
    {"inps" : ["812",       #   q
            "encoder.encoders.0.norm_final.weight",
            "encoder.encoders.0.norm_final.bias",
            ],
    "outs" : ["823"]},
    {"inps" : ["823",       #   q
            "encoder.encoders.1.norm_ff_macaron.weight",
            "encoder.encoders.1.norm_ff_macaron.bias",
            ],
    "outs" : ["834"]},
    {"inps" : ["845",       #   q
            "encoder.encoders.1.norm_mha.weight",
            "encoder.encoders.1.norm_mha.bias",
            ],
    "outs" : ["856"]},
    {"inps" : ["968",       #   q
            "encoder.encoders.1.norm_conv.weight",
            "encoder.encoders.1.norm_conv.bias",
            ],
    "outs" : ["979"]},
    {"inps" : ["1000",       #   q
            "encoder.encoders.1.norm_ff.weight",
            "encoder.encoders.1.norm_ff.bias",
            ],
    "outs" : ["1011"]},
    {"inps" : ["1022",       #   q
            "encoder.encoders.1.norm_final.weight",
            "encoder.encoders.1.norm_final.bias",
            ],
    "outs" : ["1033"]},
    {"inps" : ["1033",       #   q
            "encoder.encoders.2.norm_ff_macaron.weight",
            "encoder.encoders.2.norm_ff_macaron.bias",
            ],
    "outs" : ["1044"]},
    {"inps" : ["1055",       #   q
            "encoder.encoders.2.norm_mha.weight",
            "encoder.encoders.2.norm_mha.bias",
            ],
    "outs" : ["1066"]},
    {"inps" : ["1178",       #   q
            "encoder.encoders.2.norm_conv.weight",
            "encoder.encoders.2.norm_conv.bias",
            ],
    "outs" : ["1189"]},
    {"inps" : ["1210",       #   q
            "encoder.encoders.2.norm_ff.weight",
            "encoder.encoders.2.norm_ff.bias",
            ],
    "outs" : ["1221"]},
    {"inps" : ["1232",       #   q
            "encoder.encoders.2.norm_final.weight",
            "encoder.encoders.2.norm_final.bias",
            ],
    "outs" : ["1243"]},
    {"inps" : ["1243",       #   q
            "encoder.encoders.3.norm_ff_macaron.weight",
            "encoder.encoders.3.norm_ff_macaron.bias",
            ],
    "outs" : ["1254"]},
    {"inps" : ["1265",       #   q
            "encoder.encoders.3.norm_mha.weight",
            "encoder.encoders.3.norm_mha.bias",
            ],
    "outs" : ["1276"]},
    {"inps" : ["1388",       #   q
            "encoder.encoders.3.norm_conv.weight",
            "encoder.encoders.3.norm_conv.bias",
            ],
    "outs" : ["1399"]},
    {"inps" : ["1420",       #   q
            "encoder.encoders.3.norm_ff.weight",
            "encoder.encoders.3.norm_ff.bias",
            ],
    "outs" : ["1431"]},
    {"inps" : ["1442",       #   q
            "encoder.encoders.3.norm_final.weight",
            "encoder.encoders.3.norm_final.bias",
            ],
    "outs" : ["1453"]},
    {"inps" : ["1453",       #   q
            "encoder.encoders.4.norm_ff_macaron.weight",
            "encoder.encoders.4.norm_ff_macaron.bias",
            ],
    "outs" : ["1464"]},
    {"inps" : ["1475",       #   q
            "encoder.encoders.4.norm_mha.weight",
            "encoder.encoders.4.norm_mha.bias",
            ],
    "outs" : ["1486"]},
    {"inps" : ["1598",       #   q
            "encoder.encoders.4.norm_conv.weight",
            "encoder.encoders.4.norm_conv.bias",
            ],
    "outs" : ["1609"]},
    {"inps" : ["1630",       #   q
            "encoder.encoders.4.norm_ff.weight",
            "encoder.encoders.4.norm_ff.bias",
            ],
    "outs" : ["1641"]},
    {"inps" : ["1652",       #   q
            "encoder.encoders.4.norm_final.weight",
            "encoder.encoders.4.norm_final.bias",
            ],
    "outs" : ["1663"]},

    
    {"inps" : ["1663",       #   q
            "encoder.encoders.5.norm_ff_macaron.weight",
            "encoder.encoders.5.norm_ff_macaron.bias",
            ],
    "outs" : ["1674"]},
    {"inps" : ["1685",       #   q
            "encoder.encoders.5.norm_mha.weight",
            "encoder.encoders.5.norm_mha.bias",
            ],
    "outs" : ["1696"]},
    {"inps" : ["1808",       #   q
            "encoder.encoders.5.norm_conv.weight",
            "encoder.encoders.5.norm_conv.bias",
            ],
    "outs" : ["1819"]},
    {"inps" : ["1840",       #   q
            "encoder.encoders.5.norm_ff.weight",
            "encoder.encoders.5.norm_ff.bias",
            ],
    "outs" : ["1851"]},
    {"inps" : ["1862",       #   q
            "encoder.encoders.5.norm_final.weight",
            "encoder.encoders.5.norm_final.bias",
            ],
    "outs" : ["1873"]},

    
    {"inps" : ["1873",       #   q
            "encoder.encoders.6.norm_ff_macaron.weight",
            "encoder.encoders.6.norm_ff_macaron.bias",
            ],
    "outs" : ["1884"]},
    {"inps" : ["1895",       #   q
            "encoder.encoders.6.norm_mha.weight",
            "encoder.encoders.6.norm_mha.bias",
            ],
    "outs" : ["1906"]},
    {"inps" : ["2018",       #   q
            "encoder.encoders.6.norm_conv.weight",
            "encoder.encoders.6.norm_conv.bias",
            ],
    "outs" : ["2029"]},
    {"inps" : ["2050",       #   q
            "encoder.encoders.6.norm_ff.weight",
            "encoder.encoders.6.norm_ff.bias",
            ],
    "outs" : ["2061"]},
    {"inps" : ["2072",       #   q
            "encoder.encoders.6.norm_final.weight",
            "encoder.encoders.6.norm_final.bias",
            ],
    "outs" : ["2083"]},
    

    {"inps" : ["2083",       #   q
            "encoder.encoders.7.norm_ff_macaron.weight",
            "encoder.encoders.7.norm_ff_macaron.bias",
            ],
    "outs" : ["2094"]},
    {"inps" : ["2105",       #   q
            "encoder.encoders.7.norm_mha.weight",
            "encoder.encoders.7.norm_mha.bias",
            ],
    "outs" : ["2116"]},
    {"inps" : ["2228",       #   q
            "encoder.encoders.7.norm_conv.weight",
            "encoder.encoders.7.norm_conv.bias",
            ],
    "outs" : ["2239"]},
    {"inps" : ["2260",       #   q
            "encoder.encoders.7.norm_ff.weight",
            "encoder.encoders.7.norm_ff.bias",
            ],
    "outs" : ["2271"]},
    {"inps" : ["2282",       #   q
            "encoder.encoders.7.norm_final.weight",
            "encoder.encoders.7.norm_final.bias",
            ],
    "outs" : ["2293"]},

    
    {"inps" : ["2293",       #   q
            "encoder.encoders.8.norm_ff_macaron.weight",
            "encoder.encoders.8.norm_ff_macaron.bias",
            ],
    "outs" : ["2304"]},
    {"inps" : ["2315",       #   q
            "encoder.encoders.8.norm_mha.weight",
            "encoder.encoders.8.norm_mha.bias",
            ],
    "outs" : ["2326"]},
    {"inps" : ["2438",       #   q
            "encoder.encoders.8.norm_conv.weight",
            "encoder.encoders.8.norm_conv.bias",
            ],
    "outs" : ["2449"]},
    {"inps" : ["2470",       #   q
            "encoder.encoders.8.norm_ff.weight",
            "encoder.encoders.8.norm_ff.bias",
            ],
    "outs" : ["2481"]},
    {"inps" : ["2492",       #   q
            "encoder.encoders.8.norm_final.weight",
            "encoder.encoders.8.norm_final.bias",
            ],
    "outs" : ["2503"]},

    
    {"inps" : ["2503",       #   q
            "encoder.encoders.9.norm_ff_macaron.weight",
            "encoder.encoders.9.norm_ff_macaron.bias",
            ],
    "outs" : ["2514"]},
    {"inps" : ["2525",       #   q
            "encoder.encoders.9.norm_mha.weight",
            "encoder.encoders.9.norm_mha.bias",
            ],
    "outs" : ["2536"]},
    {"inps" : ["2648",       #   q
            "encoder.encoders.9.norm_conv.weight",
            "encoder.encoders.9.norm_conv.bias",
            ],
    "outs" : ["2659"]},
    {"inps" : ["2680",       #   q
            "encoder.encoders.9.norm_ff.weight",
            "encoder.encoders.9.norm_ff.bias",
            ],
    "outs" : ["2691"]},
    {"inps" : ["2702",       #   q
            "encoder.encoders.9.norm_final.weight",
            "encoder.encoders.9.norm_final.bias",
            ],
    "outs" : ["2713"]},

    
    {"inps" : ["2713",       #   q
            "encoder.encoders.10.norm_ff_macaron.weight",
            "encoder.encoders.10.norm_ff_macaron.bias",
            ],
    "outs" : ["2724"]},
    {"inps" : ["2735",       #   q
            "encoder.encoders.10.norm_mha.weight",
            "encoder.encoders.10.norm_mha.bias",
            ],
    "outs" : ["2746"]},
    {"inps" : ["2858",       #   q
            "encoder.encoders.10.norm_conv.weight",
            "encoder.encoders.10.norm_conv.bias",
            ],
    "outs" : ["2869"]},
    {"inps" : ["2890",       #   q
            "encoder.encoders.10.norm_ff.weight",
            "encoder.encoders.10.norm_ff.bias",
            ],
    "outs" : ["2901"]},
    {"inps" : ["2912",       #   q
            "encoder.encoders.10.norm_final.weight",
            "encoder.encoders.10.norm_final.bias",
            ],
    "outs" : ["2923"]},

    
    {"inps" : ["2923",       #   q
            "encoder.encoders.11.norm_ff_macaron.weight",
            "encoder.encoders.11.norm_ff_macaron.bias",
            ],
    "outs" : ["2934"]},
    {"inps" : ["2945",       #   q
            "encoder.encoders.11.norm_mha.weight",
            "encoder.encoders.11.norm_mha.bias",
            ],
    "outs" : ["2956"]},
    {"inps" : ["3068",       #   q
            "encoder.encoders.11.norm_conv.weight",
            "encoder.encoders.11.norm_conv.bias",
            ],
    "outs" : ["3079"]},
    {"inps" : ["3100",       #   q
            "encoder.encoders.11.norm_ff.weight",
            "encoder.encoders.11.norm_ff.bias",
            ],
    "outs" : ["3111"]},
    {"inps" : ["3122",       #   q
            "encoder.encoders.11.norm_final.weight",
            "encoder.encoders.11.norm_final.bias",
            ],
    "outs" : ["3133"]},

    
    {"inps" : ["3133",       #   q
            "encoder.after_norm.weight",
            "encoder.after_norm.bias",
            ],
    "outs" : ["encoder_out"]},
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
    tmap = graph.tensors()
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
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "layer_norm_{}".format(i)
        graph.replace_layer_norm(inputs, outputs, name)

    # for itn, itd in enumerate(div_2_mul_nodes):
    #     div_node = find_node(graph, itd)
    #     print(div_node)
    #     div_node.op = "Mul"
    #     ci = gs.Constant("Div2Mul_{}".format(itn), np.array(0.125, dtype=np.float32))
    #     div_node.inputs[1] = ci

    out_nodes = find_masked_softmax_nodes(graph)
    for i,itn in enumerate(out_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "masked_softmax_{}".format(i)
        graph.replace_masked_softmax(inputs, outputs, name)


    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()
#     graph.inputs[0].shape=[1, 16, 80]
#     graph.inputs[1].shape=[16, ]

    # That's it!
    onnx.save(gs.export_onnx(graph), output_mdl)





