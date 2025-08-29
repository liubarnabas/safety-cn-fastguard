#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两级安全分类流水线：
  - Stage 1: 预筛（二分类，Chinese-RoBERTa-wwm-ext）
  - Stage 2: 多标签（mDeBERTa-v3-base，S1–S13）
阈值从 src/inference/thresholds.json 读取；可被构造参数覆盖。
"""
import json
import os
import sys
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer

# 让脚本可直接被 import/运行（src 布局）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.prefilter_binary import Prefilter
from src.models.multilabel_mdeberta import MultiLabelMDeberta


def _load_thresholds(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认值（待校准）
    return {"theta_unsafe": 0.35, "theta_labels": 0.30,
            "labels": [f"S{i}" for i in range(1, 14)]}


class SafetyPipeline:
    def __init__(
        self,
        prefilter_path: str = "./artifacts/prefilter",
        multilabel_path: str = "./artifacts/multilabel",
        theta_unsafe: float = None,
        theta_labels: float = None,
        thresholds_path: str = "./src/inference/thresholds.json",
        device: str = None
    ):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.thresholds = _load_thresholds(thresholds_path)
        if theta_unsafe is not None:
            self.thresholds["theta_unsafe"] = float(theta_unsafe)
        if theta_labels is not None:
            self.thresholds["theta_labels"] = float(theta_labels)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Stage 1
        self.pref = Prefilter(path=prefilter_path, device=self.device)

        # Stage 2
        self.LABELS = self.thresholds.get("labels") or [f"S{i}" for i in range(1, 14)]
        self.tok_m = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        self.m = MultiLabelMDeberta()
        # 允许两种保存形式：Trainer 的 save_model 目录或仅 state_dict
        state_path = os.path.join(multilabel_path, "pytorch_model.bin")
        if os.path.exists(state_path):
            sd = torch.load(state_path, map_location="cpu")
            self.m.load_state_dict(sd)
        else:
            # 允许直接 from_pretrained（如果你用 AutoModelForSequenceClassification 训练）
            # 这里保守起见，仍走我们自定义头；若需要可改装
            pass
        self.m.eval().to(self.device)

        self.theta_unsafe = float(self.thresholds["theta_unsafe"])
        self.theta_labels = float(self.thresholds["theta_labels"])

    # -------------------
    # 单条
    # -------------------
    def classify(self, text: str) -> Dict[str, Any]:
        p_unsafe = float(self.pref.predict_proba([text])[0])
        if p_unsafe < self.theta_unsafe:
            return {"safe": True, "p_unsafe": p_unsafe, "labels": []}

        # Stage 2
        batch = self.tok_m([text], return_tensors="pt", truncation=True,
                           padding=True, max_length=384)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = self.m(**batch)["logits"]
            probs = logits.sigmoid().cpu().numpy()[0].tolist()
        labels = [self.LABELS[i] for i, pr in enumerate(probs) if pr >= self.theta_labels]
        return {
            "safe": len(labels) == 0,
            "p_unsafe": p_unsafe,
            "labels": labels,
            "probs": {self.LABELS[i]: float(pr) for i, pr in enumerate(probs)}
        }

    # -------------------
    # 批量
    # -------------------
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        # Stage 1 批预测
        p_unsafe_list = list(map(float, self.pref.predict_proba(texts)))
        # 需要进入 Stage 2 的索引
        idx2 = [i for i, p in enumerate(p_unsafe_list) if p >= self.theta_unsafe]
        out = [{"safe": True, "p_unsafe": p, "labels": []} for p in p_unsafe_list]
        if not idx2:
            return out

        texts2 = [texts[i] for i in idx2]
        batch = self.tok_m(texts2, return_tensors="pt", truncation=True,
                           padding=True, max_length=384)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = self.m(**batch)["logits"]
            probs = logits.sigmoid().cpu().numpy()

        for pos, i in enumerate(idx2):
            pr = probs[pos]
            labels = [self.LABELS[j] for j, p in enumerate(pr) if p >= self.theta_labels]
            out[i] = {
                "safe": len(labels) == 0,
                "p_unsafe": p_unsafe_list[i],
                "labels": labels,
                "probs": {self.LABELS[j]: float(p) for j, p in enumerate(pr)}
            }
        return out
