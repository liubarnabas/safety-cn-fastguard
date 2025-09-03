# tools/quantize_onnx.py
# -*- coding: utf-8 -*-
import argparse, os
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="路径：model.onnx")
    ap.add_argument("--out",  required=False, default=None, help="输出：model.int8.onnx；默认同目录")
    ap.add_argument("--s8",   action="store_true", help="使用 SignedInt8（默认 QInt8）")
    args = ap.parse_args()

    src = args.onnx
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    dst = args.out or os.path.join(os.path.dirname(src), "model.int8.onnx")
    qtype = QuantType.QInt8 if not args.s8 else QuantType.QUInt8
    quantize_dynamic(src, dst, weight_type=qtype)
    print(f"[INT8] saved -> {dst}")

if __name__ == "__main__":
    main()
