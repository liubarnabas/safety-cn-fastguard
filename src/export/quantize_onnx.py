from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("./artifacts/prefilter_onnx/model.onnx",
                 "./artifacts/prefilter_onnx/model.int8.onnx",
                 weight_type=QuantType.QInt8)
