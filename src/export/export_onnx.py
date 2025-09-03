#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a fine-tuned HF classifier to ONNX (CPU-friendly, dynamic axes).
- Ensures output dir exists
- Prints success with file size
- Optional ONNX check if onnx is installed

Usage:
  python src/export/export_onnx.py \
    --model artifacts/prefilter_deberta \
    --out   artifacts/onnx \
    --opset 17 --max-len 256
"""
import os, argparse, json, pathlib, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_prefilter_onnx():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (fine-tuned), e.g. artifacts/prefilter_deberta")
    ap.add_argument("--out",   required=True, help="Output dir for ONNX file")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    out_dir   = Path(args.out).expanduser().resolve()
    onnx_path = out_dir / "model.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "model_dir": str(model_dir),
        "out_dir": str(out_dir),
        "onnx_path": str(onnx_path),
        "opset": args.opset,
        "max_len": args.max_len
    }, ensure_ascii=False))

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval().cpu()

    # Dummy inputs (dynamic axes will generalize)
    sample = tok("测试一下ONNX导出是否成功", return_tensors="pt", max_length=args.max_len, truncation=True)
    input_ids = sample["input_ids"].cpu()
    attn_mask = sample["attention_mask"].cpu()

    # Export
    torch.onnx.export(
        model,
        (input_ids, attn_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=int(args.opset),
        do_constant_folding=True
    )

    # Post-check
    if not onnx_path.exists() or onnx_path.stat().st_size == 0:
        print("[export] ERROR: ONNX file not created.", file=sys.stderr)
        sys.exit(2)

    print(f"[export] OK -> {onnx_path}  size={onnx_path.stat().st_size/1e6:.1f} MB")

    # Optional: structural check if onnx installed
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("[export] onnx.checker: PASS")
    except Exception as e:
        print(f"[export] onnx.checker skipped or warning: {e}")

if __name__=="__main__":
    export_prefilter_onnx()
