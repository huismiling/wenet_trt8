import argparse
import ctypes
import tensorrt as trt
from pathlib import Path


def build(onnx_file,soFile,workspace=8,fp16=False,noTF32=False,verbose=False):
    engine_file = onnx_file.with_suffix(".plan")
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(str(soFile))
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(str(onnx_file))

    profile = builder.create_optimization_profile()
    if onnx_file.stem == "encoder":
        profile.set_shape("speech", (1, 16, 80), (4, 64, 80), (16, 256, 80))
        profile.set_shape("speech_lengths", (1,), (4,), (16,))
    if onnx_file.stem == "decoder":
        profile.set_shape("encoder_out", (1, 16, 256), (4, 64, 256),
                          (16, 256, 256))
        profile.set_shape("encoder_out_lens", (1,), (4,), (16,))
        profile.set_shape("hyps_pad_sos_eos", (1, 10, 64), (4, 10, 64),
                          (16, 10, 64))
        profile.set_shape("hyps_lens_sos", (1, 10), (4, 10), (16, 10))
        profile.set_shape("ctc_score", (1, 10), (4, 10), (16, 10))
    config.add_optimization_profile(profile)
    if noTF32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if "Add_" in layer.name:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                if layer.get_output_type(j) == trt.float32:
                    layer.set_output_type(j, trt.float32)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    with builder.build_serialized_network(network, config) as engine, open(str(engine_file), 'wb') as t:
        t.write(engine)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',
                        type=str,
                        default="decoder",
                        help='Choose encoder or decoder')
    parser.add_argument('--target',
                        type=str,
                        default="/target",
                        help='Git repo dir')
    parser.add_argument('--workspace',
                        type=str,
                        default="/workspace",
                        help='Git repo dir')
    parser.add_argument('--soFile',
                        type=str,
                        default="/target/SkipLayerNorm.so",
                        help='SoFile path')
    parser.add_argument('--mem',
                        type=int,
                        default=24,
                        help='Mem GiB')
    parser.add_argument('--verbose',
                    action='store_true',
                    help='Verbose log')
    parser.add_argument('--fp16',
                    action='store_true',
                    help='Build with fp16')
    parser.add_argument('--noTF32',
                    action='store_true',
                    help='Build without TF32')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    name = opt.name
    workspace = opt.mem
    fp16 = opt.fp16
    noTF32 = opt.noTF32
    verbose = opt.verbose
    soFile = opt.soFile
    assert name in ("encoder", "decoder")
    if name == "encoder":
        onnx_file = (Path(opt.target)/name).with_suffix(".onnx")
        limit = []
        build(onnx_file,soFile,workspace,fp16,noTF32,verbose)
    elif name == "decoder":
        onnx_file = (Path(opt.target)/name).with_suffix(".onnx")

        build(onnx_file,soFile,workspace,fp16,noTF32,verbose)
    else:
        raise NameError
