#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Prefilter:
    def __init__(self, name="hfl/chinese-roberta-wwm-ext", path=None, device=None):
        model_id = name if path is None else path
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, problem_type="single_label_classification"
        )
        self.model.eval().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 温度缩放参数（可选）
        self.T = 1.0
        temp_fp = os.path.join(model_id, "temperature.json")
        if os.path.exists(temp_fp):
            try:
                with open(temp_fp, "r", encoding="utf-8") as f:
                    self.T = float(json.load(f).get("T", 1.0))
            except Exception:
                self.T = 1.0

    def _softmax(self, x):
        x = x / self.T  # Temperature scaling
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / (ex.sum(axis=-1, keepdims=True) + 1e-9)

    def predict_proba(self, texts: List[str], max_len=256) -> np.ndarray:
        batch = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = self.model(**batch).logits.detach().cpu().numpy()
        probs = self._softmax(logits)
        # 输出不安全概率（类1）
        return probs[:, 1]
