# src/models/prefilter_binary.py
# -*- coding: utf-8 -*-
import os, json, numpy as np
from typing import List, Optional

class Prefilter:
    """
    通用二分类前置过滤器：
    - 自动读取 temperature.json (T) 与 theta.json (阈值)
    - 优先使用 ONNXRuntime（若存在 model.onnx），否则回退 PyTorch
    - 提供 predict_proba / predict 两种接口
    """
    def __init__(self, path: str, use_onnx: bool = True, providers: Optional[list] = None, max_len: int = 256):
        self.path = path
        self.max_len = max_len
        self.T = 1.0
        self.theta = 0.5

        # 读取温度与阈值（若无则用默认）
        try:
            with open(os.path.join(path, "temperature.json"), "r", encoding="utf-8") as f:
                self.T = float(json.load(f)["T"])
        except Exception:
            pass
        try:
            with open(os.path.join(path, "theta.json"), "r", encoding="utf-8") as f:
                self.theta = float(json.load(f)["theta_unsafe"])
        except Exception:
            pass

        # tokenizer
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=True)

        # 推理后端
        self.use_onnx = use_onnx and os.path.exists(os.path.join(path, "model.onnx"))
        if self.use_onnx:
            import onnxruntime as ort
            if providers is None:
                providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if "CUDAExecutionProvider" in ort.get_available_providers()
                             else ["CPUExecutionProvider"])
            self.ort = ort
            self.sess = ort.InferenceSession(os.path.join(path, "model.onnx"), providers=providers)
        else:
            import torch
            from transformers import AutoModelForSequenceClassification
            self.torch = torch
            self.model = AutoModelForSequenceClassification.from_pretrained(path).eval().to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def _softmax(self, x: np.ndarray, T: float):
        x = x / max(T, 1e-6)
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict_proba(self, texts: List[str], T: Optional[float] = None) -> List[float]:
        T_eff = self.T if T is None else float(T)
        outs: List[float] = []
        bs = 64
        for i in range(0, len(texts), bs):
            chunk = texts[i:i+bs]
            if self.use_onnx:
                enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="np")
                logits = self.sess.run(
                    ["logits"],
                    {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
                )[0]
                probs = self._softmax(logits, T_eff)[:, 1]
                outs.extend(probs.tolist())
            else:
                enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                with self.torch.no_grad():
                    dev = next(self.model.parameters()).device
                    enc = {k: v.to(dev) for k, v in enc.items()}
                    logits = self.model(**enc).logits.detach().float().cpu().numpy()
                probs = self._softmax(logits, T_eff)[:, 1]
                outs.extend(probs.tolist())
        return outs

    def predict(self, texts: List[str]) -> List[int]:
        probs = self.predict_proba(texts)
        th = float(self.theta)
        return [1 if p >= th else 0 for p in probs]
