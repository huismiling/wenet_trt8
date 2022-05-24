import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

mdl = onnx.load("/workspace/decoder.onnx")

mdl.graph.input[2].type.tensor_type.shape.dim[2].dim_value = 64
print(mdl.graph.input[2].type)

onnx.save(mdl, "decoder_fixed.onnx")



