import argparse
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import shape_inference

int64attr = OrderedDict([('to', 7)])
boolattr = OrderedDict([('to', 9)])
float32attr = OrderedDict([('to', 1)])


def check_gs(gs_graph):
    out_onnx = gs.export_onnx(gs_graph)
    out_onnx = shape_inference.infer_shapes(out_onnx)
    onnx.checker.check_model(out_onnx)
    return out_onnx


def deal_not(gs_graph, not_node, cast_node, where_node):
    cast_node.inputs = not_node.inputs
    not_node.inputs = cast_node.outputs
    where_node.inputs[0] = not_node.outputs[0]
    gs_graph.cleanup().toposort()
    return gs_graph


def deal_slice(gs_graph):
    not_30_save = [node for node in gs_graph.nodes if node.name == "Not_30"][0]
    slice_79_save = [
        node for node in gs_graph.nodes if node.name == "Slice_79"
    ][0]
    cast_1_output = gs.Variable("slice_in", dtype=np.int64)
    cast_1 = gs.Node(op="Cast",
                     inputs=not_30_save.outputs,
                     outputs=[cast_1_output],
                     attrs=int64attr)
    slice_79_save.inputs[0] = cast_1.outputs[0]
    gs_graph.nodes.append(cast_1)
    gs_graph.cleanup().toposort()
    return gs_graph


def deal_reshape_conv(gs_graph):
    Mul_64 = [node for node in gs_graph.nodes if node.name == "Mul_64"][0]
    Mul_64_ori_end_node = Mul_64.outputs[0]
    Mul_64.outputs.clear()
    Mul_64.inputs.clear()

    start_node = gs.Variable(name="Relu_38_output", dtype=None, shape=None)
    Relu_38 = [node for node in gs_graph.nodes if node.name == "Relu_38"][0]
    Relu_38.outputs[0] = start_node

    MatMul_61 = [node for node in gs_graph.nodes
                 if node.name == "MatMul_61"][0]
    conv_weight = np.transpose(MatMul_61.inputs[1].values, (1, 0)).reshape(
        (-1, 256, 1, 19)) * 16
    conv_weight = gs.Constant(name="ReshapeConv1_weight,", values=conv_weight)

    Add_62 = [node for node in gs_graph.nodes if node.name == "Add_62"][0]
    conv_bias = Add_62.inputs[0].values * 16
    conv_bias = gs.Constant(name="ReshapeConv1_bias,", values=conv_bias)

    Conv_37 = [node for node in gs_graph.nodes if node.name == "Conv_37"][0]
    newConv = Conv_37.copy()

    newConv.attrs['kernel_shape'] = [1, 19]
    newConv.attrs['strides'] = [1, 1]
    newConv.name = "ReshapeConv1"

    newConv.inputs = [start_node, conv_weight, conv_bias]

    end_node = Add_62.outputs[0]
    Add_62.outputs.clear()
    newConv.outputs = [end_node]

    Mul_64_output = gs.Variable(name="Mul_64_output", dtype=None, shape=None)
    Mul_64.outputs = [Mul_64_output]

    Transpose_51 = [
        node for node in gs_graph.nodes if node.name == "Transpose_51"
    ][0]
    Transpose_51.inputs.clear()
    Transpose_51.outputs.clear()

    ReshapeConv1Transpose_param = OrderedDict([('perm', [0, 2, 1, 3])])
    ReshapeConv1Transpose_output = gs.Variable(
        name="ReshapeConv1Transpose_output", dtype=None, shape=None)

    ReshapeConv1Transpose = gs.Node(op="Transpose",
                                    name="ReshapeConv1Transpose",
                                    inputs=[end_node],
                                    outputs=[ReshapeConv1Transpose_output],
                                    attrs=ReshapeConv1Transpose_param)
    ReshapeConv1Unsqueeze_param = gs.Constant(
        name="ReshapeConv1Unsqueeze_param",
        values=np.array([-1], dtype=np.int64))
    ReshapeConv1Unsqueeze = gs.Node(
        name="ReshapeConv1Unsqueeze",
        op="Squeeze",
        inputs=[ReshapeConv1Transpose_output, ReshapeConv1Unsqueeze_param],
        outputs=[Mul_64_ori_end_node])
    nodes_temp = [newConv, ReshapeConv1Transpose, ReshapeConv1Unsqueeze]
    for node in nodes_temp:
        gs_graph.nodes.append(node)
    gs_graph.cleanup().toposort()
    return gs_graph


def deal_Slice_74(gs_graph):

    Slice_74 = [node for node in gs_graph.nodes if node.name == "Slice_74"][0]
    Slice_74_outputs = gs.Variable(name="Slice_74_outputs",
                                   dtype=None,
                                   shape=None)
    Slice_74.outputs[0] = Slice_74_outputs

    Shape_609 = [node for node in gs_graph.nodes
                 if node.name == "Shape_609"][0]
    Shape_609.inputs.clear()
    Shape_609.outputs.clear()

    def chang_reshape_inputs(gs_graph, name):
        node = [node for node in gs_graph.nodes if node.name == name][0]
        MatMul_node_outpts = node.o().outputs[0]
        node.o().outputs.clear()
        MatMul = node.inputs[0]
        MatMul_node = MatMul.inputs[0]
        MatMul_weight = MatMul_node.inputs[1].values

        MatMul_node.outputs.clear()
        MatMul_node.inputs.clear()
        return MatMul_weight, MatMul_node_outpts

    name_list = [
        "Reshape_145", "Reshape_302", "Reshape_459", "Reshape_616",
        "Reshape_773", "Reshape_930", "Reshape_1087", "Reshape_1244",
        "Reshape_1401", "Reshape_1558", "Reshape_1715", "Reshape_1872"
    ]

    Concat_615 = [
        node for node in gs_graph.nodes if node.name == "Concat_615"
    ][0]
    Concat_615_param0 = gs.Constant(name="Concat_615_param0",
                                    values=np.array([len(name_list) * 64],
                                                    dtype=np.int64))
    Concat_615.inputs[-1] = Concat_615_param0

    MatMul_weight_node_outpts_list = [
        chang_reshape_inputs(gs_graph, name) for name in name_list
    ]

    MatMul_weight_list = [value[0] for value in MatMul_weight_node_outpts_list]
    Slice_74_weight_value = Slice_74.inputs[0].values

    Slice_74_weight_value_list = []
    for MatMul_weight in MatMul_weight_list:
        Slice_74_weight_value_list.append(
            np.dot(Slice_74_weight_value, MatMul_weight).reshape(
                (1, 5000, 4, 64)).transpose((0, 2, 3, 1)))
    Slice_74_weight_value = np.concatenate(Slice_74_weight_value_list, 2)

    Slice_74_weight = gs.Constant(name="Slice_74_weight",
                                  values=Slice_74_weight_value)
    Slice_74.inputs[0] = Slice_74_weight
    Slice_74.inputs[3] = gs.Constant(name="Slice_74_axis",
                                     values=np.array([3], dtype=np.int64))

    Reshape_616 = [
        node for node in gs_graph.nodes if node.name == "Reshape_616"
    ][0]
    Reshape_616.inputs[0] = Slice_74_outputs

    Transpose_623 = [
        node for node in gs_graph.nodes if node.name == "Transpose_623"
    ][0]
    Transpose_623.outputs.clear()
    MatMul_outputs_list = [
        value[1] for value in MatMul_weight_node_outpts_list
    ]
    for i, MatMul_outputs in enumerate(MatMul_outputs_list):
        Slice_node = [
            node for node in gs_graph.nodes if node.name == "Slice_74"
        ][0].copy()
        Slice_node_param = [
            gs.Constant(name="MatMul_Slice_{}_starts".format(i),
                        values=np.array([i * 64], dtype=np.int32)),
            gs.Constant(name="MatMul_Slice_{}_ends".format(i),
                        values=np.array([(i + 1) * 64], dtype=np.int32)),
            gs.Constant(name="MatMul_Slice_{}_axes".format(i),
                        values=np.array([2], dtype=np.int32)),
            gs.Constant(name="MatMul_Slice_{}_steps".format(i),
                        values=np.array([1], dtype=np.int32)),
        ]
        Slice_node.inputs = [Slice_74_outputs] + Slice_node_param
        Slice_node.outputs = [MatMul_outputs]
        Slice_node.name = "MatMul_Slice_{}".format(i)

        gs_graph.nodes.append(Slice_node)
    gs_graph.cleanup().toposort()
    return gs_graph


def encoder(opt):
    sim = opt.sim
    dynamic = opt.dynamic
    name = "encoder"

    target = Path(opt.target)
    workspace = Path(opt.workspace)

    onnx_file = (workspace / name).with_suffix(".onnx")
    save_onnx_dir = (target / name).with_suffix(".onnx")

    onnx_graph = onnx.load(str(onnx_file))
    gs_graph = gs.import_onnx(onnx_graph)

    gs_graph.fold_constants()

    not2deal = [
        193, 204, 350, 361, 507, 518, 664, 675, 821, 832, 978, 989, 1135, 1146,
        1292, 1303, 1449, 1460, 1606, 1617, 1763, 1774, 1920, 1931
    ]
    cast2deal = [
        194, 206, 351, 363, 508, 520, 665, 677, 822, 834, 979, 991, 1136, 1148,
        1293, 1305, 1450, 1462, 1607, 1619, 1764, 1776, 1921, 1933
    ]
    where2deal = [
        196, 208, 353, 365, 510, 522, 667, 679, 824, 836, 981, 993, 1138, 1150,
        1295, 1307, 1452, 1464, 1609, 1621, 1766, 1778, 1923, 1935
    ]

    len2deal = len(not2deal)

    gs_graph = deal_slice(gs_graph)

    for i in range(len2deal):
        not_node = [
            node for node in gs_graph.nodes
            if node.name == f"Not_{not2deal[i]}"
        ][0]
        cast_node = [
            node for node in gs_graph.nodes
            if node.name == f"Cast_{cast2deal[i]}"
        ][0]
        where_node = [
            node for node in gs_graph.nodes
            if node.name == f"Where_{where2deal[i]}"
        ][0]
        gs_graph = deal_not(gs_graph, not_node, cast_node, where_node)

    gs_graph = deal_reshape_conv(gs_graph)
    gs_graph = deal_Slice_74(gs_graph)
    for node in gs_graph.nodes:
        if node.name == "Gather_67":
            nextnode = node.o()
            nnextnode = node.o().o()
            nnextnode.inputs[0] = node.outputs[0]
            nextnode.outputs.clear()
    gs_graph.cleanup().toposort()
    onnx_model = check_gs(gs_graph)
    gs_graph = gs.import_onnx(onnx_model)

    gs_graph.fold_constants(fold_shapes=True)
    gs_graph.cleanup().toposort()
    onnx_model = gs.export_onnx(gs_graph)
    if sim:
        try:
            import onnxsim
            shape_dict = None
            if dynamic:
                print("Use dynamic inputs")
                shape_dict = {
                    "speech": [4, 64, 80],
                    "speech_lengths": [
                        4,
                    ]
                }
            onnx_model, check = onnxsim.simplify(onnx_model,
                                                 dynamic_input_shape=dynamic,
                                                 input_shapes=shape_dict)
            assert check, 'assert check failed'
        except Exception as e:
            print(f"onnx-simplifier failed: message\n{e}")

    onnx.save(onnx_model, str(save_onnx_dir))


def decoder(opt):
    sim = opt.sim
    dynamic = opt.dynamic
    name = "decoder"
    target = Path(opt.target)
    workspace = Path(opt.workspace)

    onnx_file = (workspace / name).with_suffix(".onnx")
    save_onnx_dir = (target / name).with_suffix(".onnx")

    onnx_graph = onnx.load(str(onnx_file))

    if sim:
        try:
            import onnxsim
            shape_dict = None
            if dynamic:
                print("Use dynamic inputs")
                shape_dict = {
                    "encoder_out": [4, 64, 256],
                    "encoder_out_lens": [
                        4,
                    ],
                    "hyps_pad_sos_eos": [4, 10, 64],
                    "hyps_lens_sos": [4, 10],
                    "ctc_score": [4, 10],
                }

            onnx_graph, check = onnxsim.simplify(onnx_graph,
                                                 dynamic_input_shape=dynamic,
                                                 input_shapes=shape_dict)
            assert check, 'assert check failed'
        except Exception as e:
            print(f"onnx-simplifier failed: message\n{e}")

    gs_graph = gs.import_onnx(onnx_graph)
    # gs_graph.fold_constants()

    reducesum_1083 = [
        node for node in gs_graph.nodes if node.name == "ReduceSum_1083"
    ][0]
    reducesum_1083.attrs = OrderedDict([('keepdims', 1)])
    slice_164 = [node for node in gs_graph.nodes
                 if node.name == "Slice_164"][0]
    slice_164.inputs[0].values = slice_164.inputs[0].values[:, :63, :]
    add_167 = [node for node in gs_graph.nodes if node.name == "Add_167"][0]
    add_167.inputs[1] = slice_164.inputs[0]
    gs_graph.cleanup().toposort()
    onnx_graph = check_gs(gs_graph)
    gs_graph = gs.import_onnx(onnx_graph)

    encoder_out_shape = gs.Variable("encoder_out_shape_0",
                                    dtype=np.int64,
                                    shape=[
                                        3,
                                    ])
    shape_00 = gs.Node(op="Shape",
                       name='encoder_out_shape',
                       inputs=[gs_graph.inputs[0]],
                       outputs=[encoder_out_shape])
    gs_graph.nodes.append(shape_00)
    gather_2 = [node for node in gs_graph.nodes if node.name == "Gather_2"][0]
    gather_5 = [node for node in gs_graph.nodes if node.name == "Gather_5"][0]
    gather_8 = [node for node in gs_graph.nodes if node.name == "Gather_8"][0]
    gather_2.inputs[0] = encoder_out_shape
    gather_5.inputs[0] = encoder_out_shape
    gather_8.inputs[0] = encoder_out_shape

    # 20220422 改动
    Tile_14 = [node for node in gs_graph.nodes if node.name == "Tile_14"][0]
    Tile_14.inputs[0] = gs_graph.inputs[0]
    Unsqueeze_31 = [
        node for node in gs_graph.nodes if node.name == "Unsqueeze_31"
    ][0]
    Unsqueeze_31.inputs[0] = gather_2.outputs[0]
    Slice_87 = [node for node in gs_graph.nodes if node.name == "Slice_87"][0]
    Slice_87.inputs[2].values = np.array([64])
    gather_154 = [
        node for node in gs_graph.nodes if node.name == "Gather_154"
    ][0]
    gather_154.inputs[0].values = gather_154.inputs[0].values[:1000, :]

    gs_graph = bignot2less(gs_graph)
    # gs_graph = div2mul(gs_graph,[487])
    gs_graph.cleanup().toposort()
    onnx_graph = check_gs(gs_graph)

    if sim:
        try:
            import onnxsim
            shape_dict = None
            if dynamic:

                shape_dict = {
                    "encoder_out": [4, 64, 256],
                    "encoder_out_lens": [
                        4,
                    ],
                    "hyps_pad_sos_eos": [4, 10, 64],
                    "hyps_lens_sos": [4, 10],
                    "ctc_score": [4, 10],
                }

            onnx_graph, check = onnxsim.simplify(onnx_graph,
                                                 dynamic_input_shape=dynamic,
                                                 input_shapes=shape_dict)
            assert check, 'assert check failed'
        except Exception as e:
            print(f"onnx-simplifier failed: message\n{e}")
    onnx.save(onnx_graph, str(save_onnx_dir))

def bignot2less(gs_graph):
    i = 0
    for node in gs_graph.nodes:
        if node.op == "GreaterOrEqual":
            if node.o().op == "Not":
                tmp = gs.Variable(name=f"Add_Out{i}")
                i+=1
                Not = node.o()
                inp = node.inputs
                out = [tmp]
                Less = gs.Node(op="Less",
                                name=f"Less_Add{i}",
                                inputs=inp,
                                outputs=out)
                i+=1
                gs_graph.nodes.append(Less)
                nextNode,nnextNode = Not.o(0),Not.o(1)
                nextNode.inputs[0] = tmp
                nnextNode.inputs[0] = tmp
                node.outputs.clear()
                Not.outputs.clear()
            elif node.o().op == "Unsqueeze":
                Unsqueeze = node.o()
                if node.o().o().op == "Not":
                    tmp = gs.Variable(name=f"Add_Out{i}")
                    i+=1
                    Not = Unsqueeze.o()
                    inp = node.inputs
                    out = [tmp]
                    Less = gs.Node(op="Less",
                                   name=f"Less_Add{i}",
                                   inputs=inp,
                                   outputs=out)
                    i += 1
                    gs_graph.nodes.append(Less)
                    Unsqueeze.inputs[0] = tmp
                    Unsqueeze.outputs[0] = Not.o().inputs[0] # only for decoder
                    node.outputs.clear()
                    Not.outputs.clear()
    return gs_graph

def div2mul(gs_graph,nameList=[]):
    if nameList:
        for node in gs_graph.nodes:
            if node.op == "Div":
                index = int(node.name.split("_")[1])
                if index in nameList: # [156, 313, 470, 627, 784, 941, 1098, 1255, 1412, 1569, 1726, 1883]
                    if isinstance(node.inputs[1],gs.ir.tensor.Constant):
                        nameList.remove(index)
                        node.op = "Mul"
                        node.name = f"Mul_Add{index}"
                        value = 1.0/node.inputs[1].values
                        value = np.array(value).astype(np.float32)
                        name = node.inputs[1].name
                        tmp = gs.Constant(name=f"DIV2_{name}_{random.randint(1,100)}",values=value)
                        node.inputs[1] = tmp
    return gs_graph



def register_ln(opt,name,skip=False):
    nLayerNormPlugin = 0
    target = Path(opt.target)
    onnx_file = (target / name).with_suffix(".onnx")
    gs_graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(str(onnx_file))))
    for node in gs_graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1):

            inputTensor = node.inputs[0]
            if not skip:
                lastDivNode = node.o().o(0).o().o().o().o()
                layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=[inputTensor], outputs=[lastDivNode.outputs[0]])
                gs_graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1
                lastDivNode.outputs.clear()
                continue
            else:
                MulNode = node.o().o(1).o()
                AddNode = node.o().o(1).o().o() # lastDivNode
                beta, gamma = AddNode.inputs[1].values, MulNode.inputs[1].values
                mLd = np.array(256)
                plugin_version = "2"
                type_id = np.array(1) # 0:fp32 1:fp16 2:int8
                attrs = OrderedDict(
                    beta=gs.Constant(name='beta', values=beta),
                    gamma=gs.Constant(name='gamma', values=gamma),
                    mLd=gs.Constant(name="mLd", values=mLd),
                    plugin_version=plugin_version,
                    type_id=gs.Constant(name="type_id", values=type_id)
                )
                layerNormN = gs.Node(
                    name="SkipLayerNormN-" + str(nLayerNormPlugin),
                    op="MySkipLNPluginDynamic",
                    inputs=[inputTensor],
                    outputs=[AddNode.outputs[0]],
                    attrs=attrs
                )
                gs_graph.nodes.append(layerNormN)
                nLayerNormPlugin += 1
                AddNode.outputs.clear()
                continue

    gs_graph.cleanup().toposort()
    onnx.save(gs.export_onnx(gs_graph), str(onnx_file))
    return onnx_file.name


def DIV2MUL(opt,name,nameList):
    target = Path(opt.target)
    onnx_file = (target / name).with_suffix(".onnx")
    gs_graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(str(onnx_file))))
    gs_graph = div2mul(gs_graph,nameList)
    gs_graph.cleanup().toposort()
    onnx.save(gs.export_onnx(gs_graph), str(onnx_file))
    return onnx_file.name
    
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        default="all",
                        help='Choose encoder or decoder')
    parser.add_argument('--target',
                        type=str,
                        default="/target",
                        help='Git repo dir')
    parser.add_argument('--workspace',
                        type=str,
                        default="/workspace",
                        help='Git repo dir')
    parser.add_argument('--dynamic',
                        action='store_true',
                        help='Use dunamic inputs')
    parser.add_argument('--ln',
                    action='store_true',
                    help='Use layer norm')
    parser.add_argument('--skip',
                    action='store_true',
                    help='Use skiplayer norm')
    parser.add_argument('--sim', action='store_true', help='Simply onnx')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    name = opt.name
    ln = opt.ln
    assert name in ("encoder", "decoder", "all")
    if name == "encoder":
        encoder(opt)
    elif name == "decoder":
        decoder(opt)
    elif name == "all":
        encoder(opt)
        decoder(opt)
    else:
        raise NameError
    if ln:
        print("Register layernorm in onnxs")
        if name == "encoder":
            nameList = [156, 313, 470, 627, 784, 941, 1098, 1255, 1412, 1569, 1726, 1883]
            name = register_ln(opt,name,opt.skip)
            name = DIV2MUL(opt,name,nameList)
            print(f"Finish register layer norm for {name}!")
        elif name == "decoder":
            nameList = [205, 267, 346, 408, 487, 549, 628, 690, 769, 831, 910, 972]
            name = register_ln(opt,name,opt.skip)
            name = DIV2MUL(opt,name,nameList)
            print(f"Finish register layer norm for {name}!")
        elif name == "all":
            nameList = [156, 313, 470, 627, 784, 941, 1098, 1255, 1412, 1569, 1726, 1883]
            name = register_ln(opt,"encoder",opt.skip)
            name = DIV2MUL(opt,name,nameList)
            print(f"Finish register layer norm for {name}!")
            nameList = [205, 267, 346, 408, 487, 549, 628, 690, 769, 831, 910, 972]
            name = register_ln(opt,"decoder",opt.skip)
            name = DIV2MUL(opt,name,nameList)
            print(f"Finish register layer norm for {name}!")