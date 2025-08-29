#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChineseSafe 基准评测与阈值校准脚本
支持两种模式：
  1) prefilter  —— 仅对 Stage-1 预筛模型（Chinese-RoBERTa 二分类）做阈值网格搜索
  2) pipeline   —— 对两级流水线整体做二分类评测（若找不到多标签模型，则自动回退为仅预筛）

数据集：SUSTech/ChineseSafe
字段说明（以当前公开版本为准）：
  - text:    待判定文本
  - label:   "违规" / "不违规"（二分类真值）
  - subject: 主题类别（如“淫秽色情”“政治错误”等，11 个中文主题）

注意：ChineseSafe 不含我们训练用的 S1–S13 字段，因此此处评测目标为“是否违规（二分类）”。
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
from datasets import load_dataset

# ---------------------------
# 让脚本可直接运行：把项目根目录加入 sys.path
#（src/ 布局下，脚本作为文件运行时默认找不到顶层包；见 Python 模块搜索路径规范）
# ---------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 为了让脚本独立可运行，这里自带简单度量函数（避免依赖项目内其它 metrics 文件）
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
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "support_pos": int((y_true == 1).sum()),
        "support_neg": int((y_true == 0).sum()),
    }

def load_chinesesafe(max_samples=-1, slice_prefix="test"):
    """
    载入 ChineseSafe 测试集；返回 texts, y_true（二分类）
    y_true: 1 表示 违规, 0 表示 不违规
    """
    split = slice_prefix if max_samples < 0 else f"{slice_prefix}[:{max_samples}]"
    ds = load_dataset("SUSTech/ChineseSafe", split=split)

    def is_violation(lbl):
        if isinstance(lbl, str):
            return 1 if lbl.strip() == "违规" else 0
        try:
            return 1 if int(lbl) == 1 else 0
        except Exception:
            return 0

    texts = [ex["text"] for ex in ds]
    y = [is_violation(ex["label"]) for ex in ds]
    return texts, y

def _safe_import_prefilter(prefilter_path):
    from src.models.prefilter_binary import Prefilter
    return Prefilter(path=prefilter_path)

def _safe_import_pipeline(theta_unsafe, theta_labels, prefilter_path, multilabel_path):
    from src.inference.pipeline import SafetyPipeline
    return SafetyPipeline(
        prefilter_path=prefilter_path,
        multilabel_path=multilabel_path,
        theta_unsafe=theta_unsafe,
        theta_labels=theta_labels
    )

def mode_prefilter(args):
    texts, y_true = load_chinesesafe(max_samples=args.max_samples)
    pre = _safe_import_prefilter(args.prefilter_path)

    # 计算所有样本的 p_unsafe
    probs = []
    batch_size = args.batch_size
    for i in range(0, len(texts), batch_size):
        probs.extend(pre.predict_proba(texts[i:i + batch_size]))
    probs = list(map(float, probs))

    # 阈值网格
    start, end, step = args.grid
    thresholds = np.arange(start, end + 1e-12, step)
    best = {"theta": None, "f1": -1.0}
    print(f"[prefilter] candidates: theta in [{start:.2f}, {end:.2f}] step={step:.2f}, samples={len(texts)}")
    for th in thresholds:
        y_pred = [1 if p >= th else 0 for p in probs]
        m = bin_metrics(y_true, y_pred)
        print(f"theta_unsafe={th:.2f}  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}  "
              f"Acc={m['accuracy']:.3f}  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
        if m["f1"] > best["f1"]:
            best = {"theta": float(th), "f1": float(m["f1"]), "metrics": m}

    # 汇总
    print("\n[prefilter][BEST] theta_unsafe={theta:.2f}  F1={f1:.3f}".format(**best))
    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump({"mode": "prefilter", "best": best}, f, ensure_ascii=False, indent=2)
        print(f"[prefilter] saved report to: {args.report_file}")

def mode_pipeline(args):
    texts, y_true = load_chinesesafe(max_samples=args.max_samples)

    # 优先尝试加载完整流水线，失败则回退为仅预筛
    try:
        pipe = _safe_import_pipeline(args.theta_unsafe, args.theta_labels,
                                     args.prefilter_path, args.multilabel_path)
        use_prefilter_only = False
    except Exception as e:
        print(f"[pipeline] WARN: failed to load multilabel model, fallback to prefilter-only. err={e}")
        pre = _safe_import_prefilter(args.prefilter_path)
        use_prefilter_only = True

    y_pred = []
    batch_size = args.batch_size
    if use_prefilter_only:
        probs = []
        for i in range(0, len(texts), batch_size):
            probs.extend(pre.predict_proba(texts[i:i + batch_size]))
        y_pred = [1 if float(p) >= args.theta_unsafe else 0 for p in probs]
    else:
        # 两级流水线：预筛+多标签；只要最终判为 unsafe（包含任一 S 标签），就预测 1
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            for t in chunk:
                r = pipe.classify(t)
                y_pred.append(0 if r.get("safe", True) else 1)

    m = bin_metrics(y_true, y_pred)
    print(f"[pipeline] theta_unsafe={args.theta_unsafe:.2f}, theta_labels={args.theta_labels:.2f}, "
          f"samples={len(texts)}")
    print("P={precision:.3f} R={recall:.3f} F1={f1:.3f} Acc={accuracy:.3f}  "
          "TP={tp} FP={fp} FN={fn} TN={tn}".format(**m))

    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            json.dump({
                "mode": "pipeline",
                "theta_unsafe": args.theta_unsafe,
                "theta_labels": args.theta_labels,
                "metrics": m
            }, f, ensure_ascii=False, indent=2)
        print(f"[pipeline] saved report to: {args.report_file}")

def parse_grid(s: str):
    """
    解析类似 "0.25,0.45,0.05" 的网格字符串
    """
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--grid must be 'start,end,step'")
    start, end, step = parts
    if step <= 0 or end < start:
        raise ValueError("grid: invalid range")
    return start, end, step

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prefilter", "pipeline"], required=True)
    p.add_argument("--prefilter-path", default="./artifacts/prefilter")
    p.add_argument("--multilabel-path", default="./artifacts/multilabel")
    p.add_argument("--theta-unsafe", type=float, default=0.35)
    p.add_argument("--theta-labels", type=float, default=0.30)
    p.add_argument("--grid", type=parse_grid, default=(0.25, 0.45, 0.05))
    p.add_argument("--max-samples", type=int, default=2000,
                   help="-1 表示用完整 test 集（~20k），否则用前 N 条加速调参")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--report-file", default="./artifacts/eval/chinesesafe_report.json")
    args = p.parse_args()

    # 一些环境变量的小优化（非必须）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.mode == "prefilter":
        mode_prefilter(args)
    else:
        mode_pipeline(args)

if __name__ == "__main__":
    main()
