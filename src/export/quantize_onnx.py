#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantize ONNX (dynamic INT8) with recommended pre-processing.

Usage:
  # 已有 model.pre.onnx
  python src/export/quantize_onnx.py \
    --model artifacts/onnx/model.pre.onnx \
    --out   artifacts/onnx/model.int8.onnx

  # 也可直接喂 model.onnx（脚本会自动预处理成 .pre.onnx 再量化）
  python src/export/quantize_onnx.py \
    --model artifacts/onnx/model.onnx \
    --out   artifacts/onnx/model.int8.onnx
"""
import argparse
from pathlib import Path
import sys

from onnxruntime.quantization import quantize_dynamic, QuantType
# 预处理 API 在 shape_inference 模块里
from onnxruntime.quantization.shape_inference import quant_pre_process

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX 输入路径（.pre.onnx 或 .onnx）")
    ap.add_argument("--out",   required=True, help="量化后 INT8 ONNX 输出路径")
    # 可选回退：极端情况下可跳过某些步骤
    ap.add_argument("--skip-optimization", action="store_true", help="预处理时跳过图优化（>2GB 大模型可用）")
    ap.add_argument("--skip-onnx-shape",   action="store_true", help="预处理时跳过 ONNX 形状推断")
    ap.add_argument("--skip-symbolic-shape", action="store_true", help="预处理时跳过符号形状推断")
    args = ap.parse_args()

    in_path  = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 决定预处理输入/输出
    #    - 如果用户直接传了 *.pre.onnx 且存在：直接量化
    #    - 如果传了 *.pre.onnx 但不存在：尝试同目录下的 *.onnx 做预处理
    #    - 如果传了 *.onnx ：做预处理
    if in_path.suffix == ".onnx" and in_path.name.endswith(".pre.onnx"):
        # 罕见：文件名以 .pre.onnx 结尾但后缀判断仍是 .onnx，这里统一当作 pre 文件
        is_pre = True
    else:
        is_pre = in_path.name.endswith(".pre.onnx")

    if is_pre:
        if not in_path.exists():
            # 尝试从同目录的非 pre 文件回退
            base = in_path.name.replace(".pre.onnx", ".onnx")
            candidate = in_path.with_name(base)
            if not candidate.exists():
                print(f"[quantize] ERROR: {in_path} 不存在，且未找到回退源 {candidate}", file=sys.stderr)
                sys.exit(2)
            # 需要预处理：candidate -> in_path
            print(f"[quantize] preprocessing (recover missing pre) -> {in_path}")
            quant_pre_process(
                str(candidate),                     # input_model_path
                str(in_path),                       # output_model_path
                bool(args.skip_optimization),       # skip_optimization
                bool(args.skip_onnx_shape),         # skip_onnx_shape
                bool(args.skip_symbolic_shape),     # skip_symbolic_shape
            )
        # 走量化
        pre_model = in_path
    else:
        # 需要预处理：in_path.onnx -> pre_out
        pre_model = in_path.with_suffix(".pre.onnx")
        print(f"[quantize] preprocessing -> {pre_model}")
        quant_pre_process(
            str(in_path),                          # input_model_path
            str(pre_model),                        # output_model_path
            bool(args.skip_optimization),          # skip_optimization
            bool(args.skip_onnx_shape),            # skip_onnx_shape
            bool(args.skip_symbolic_shape),        # skip_symbolic_shape
        )

    # 2) 动态量化（INT8）
    print(f"[quantize] dynamic INT8 -> {out_path}")
    quantize_dynamic(
        model_input=str(pre_model),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
        optimize_model=False  # 图优化已在预处理阶段完成
    )
    print("[quantize] done.")

if __name__ == "__main__":
    main()
