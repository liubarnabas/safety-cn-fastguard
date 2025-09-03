#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChineseSafe 基准评测与阈值校准脚本
新增 --score 指标选择：accuracy/f1/precision/recall （默认 accuracy）
自动读取训练阶段保存的 temperature.json (T) 与 theta.json (θ)；可在网格上重新搜索更优 θ。
"""
import argparse
import json
import os
import sys
from typing import List

import numpy as np
from datasets import load_dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def bin_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def load_chinesesafe(max_samples=-1, slice_prefix="test"):
    split = slice_prefix if max_samples < 0 else f"{slice_prefix}[:{max_samples}]"
    ds = load_dataset("SUSTech/ChineseSafe", split=split)

    def is_violation(lbl):
        # 数值优先
        try:
            return 1 if int(lbl) == 1 else 0
        except Exception:
            pass
        s = str(lbl).strip().lower()
        if any(k in s for k in ["违", "不安全", "有害", "仇恨", "歧视", "冒犯", "辱骂", "毒", "暴力", "极端", "恐怖", "诈骗", "色情", "自杀", "自残"]):
            return 1
        return 0

    texts = [ex["text"] for ex in ds]
    y = [is_violation(ex["label"]) for ex in ds]
    return texts, y

def _import_prefilter(prefilter_path):
    from src.models.prefilter_binary import Prefilter
    return Prefilter(path=prefilter_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefilter-path", default="./artifacts/prefilter")
    p.add_argument("--grid", type=str, default="0.05,0.95,0.01")
    p.add_argument("--max-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--score", choices=["accuracy","f1","precision","recall"], default="accuracy")
    p.add_argument("--report-file", default="./artifacts/eval/chinesesafe_prefilter_report.json")
    args = p.parse_args()

    start, end, step = [float(x) for x in args.grid.split(",")]
    texts, y_true = load_chinesesafe(max_samples=args.max_samples)

    # 读取训练时保存的 T 和 θ（若无则 T=1.0，θ=0.5）
    T, theta_default = 1.0, 0.5
    try:
        with open(os.path.join(args.prefilter_path, "temperature.json"), "r", encoding="utf-8") as f:
            T = float(json.load(f)["T"])
    except Exception:
        pass
    try:
        with open(os.path.join(args.prefilter_path, "theta.json"), "r", encoding="utf-8") as f:
            theta_default = float(json.load(f)["theta_unsafe"])
    except Exception:
        pass

    pre = _import_prefilter(args.prefilter_path)

    probs = []
    for i in range(0, len(texts), args.batch_size):
        probs.extend(pre.predict_proba(texts[i:i+args.batch_size], T=T))
    probs = list(map(float, probs))

    grid = np.arange(start, end + 1e-12, step)
    best = {"theta": theta_default, "metric": -1.0, "metrics": {}}
    print(f"[prefilter] score={args.score}, theta in [{start:.2f},{end:.2f}] step={step:.2f}, samples={len(texts)}, T={T:.3f}")
    for th in grid:
        y_pred = [1 if p >= th else 0 for p in probs]
        m = bin_metrics(y_true, y_pred)
        print(f"theta_unsafe={th:.2f}  Acc={m['accuracy']:.3f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        metric_val = m[args.score]
        if metric_val > best["metric"]:
            best = {"theta": float(th), "metric": float(metric_val), "metrics": m}

    print(f"\n[prefilter][BEST-{args.score}] theta_unsafe={best['theta']:.2f}  {args.score}={best['metric']:.3f}")
    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump({"mode":"prefilter","score":args.score,"best":best,"T":T}, f, ensure_ascii=False, indent=2)
        print(f"[prefilter] saved report: {args.report_file}")

if __name__ == "__main__":
    main()
