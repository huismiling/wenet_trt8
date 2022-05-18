import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

mdl = onnx.load("/workspace/decoder.onnx")

mdl.graph.input[2].type.tensor_type.shape.dim[2].dim_value = 64
print(mdl.graph.input[2].type)

ext_out = []
for itn in ["377", "476", "575", "693", "792", "910", "1009", 
            "1127", "1226", "1344", "1443", "1561", "1660",  # 12+2 
            "388", "461", "460", "443", "442", "409", "393", "357", "260"]:
    if itn in ["357", "260"]:
        ext_out.append(helper.make_tensor_value_info(itn, TensorProto.BOOL, None))
    else:
        ext_out.append(helper.make_tensor_value_info(itn, TensorProto.FLOAT, None))

mdl.graph.output.extend(ext_out)

onnx.save(mdl, "decoder_fixed.onnx")



