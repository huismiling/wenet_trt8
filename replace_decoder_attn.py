

import onnx 
import onnx_graphsurgeon as gs





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




cross_attn_nodes = [
        {"inps" : ["476",       #   q
                "214",          #   enc_in
                "encoder_out_lens",      #   mask
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
                "encoder_out_lens",      #   mask
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
                "encoder_out_lens",      #   mask
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
                "encoder_out_lens",      #   mask
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
                "encoder_out_lens",      #   mask
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
                "encoder_out_lens",      #   mask
                "1953", "decoder.decoders.5.src_attn.linear_q.bias",
                "1957", "decoder.decoders.5.src_attn.linear_k.bias", 
                "1961", "decoder.decoders.5.src_attn.linear_v.bias",
                "1967", "decoder.decoders.5.src_attn.linear_out.bias",
                "decoder.decoders.5.norm2.weight",
                "decoder.decoders.5.norm2.bias",
                ],
        "outs" : ["1660"]},
    ]
if __name__ == "__main__":
    graph = gs.import_onnx(onnx.load("decoder_fixed.onnx"))

    tmap = graph.tensors()
    # You can figure out the input and output tensors using Netron. In our case:
    # Inputs: [inp, MIN_VAL, MAX_VAL]
    # Outputs: [max_out]
    for i,itn in enumerate(cross_attn_nodes):
        inputs = [tmap[i] for i in itn["inps"]]
        outputs = [tmap[i] for i in itn["outs"]]
        name = "cross_attn_{}".format(i)
        attrs = {"AttentionType":"cross"}
        graph.replace_attn(inputs, outputs, name, attrs)

    # Remove the now-dangling subgraph.
    graph.cleanup().toposort()

    # That's it!
    onnx.save(gs.export_onnx(graph), "decoder_replaced.onnx")





