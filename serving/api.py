#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.prefilter_binary import Prefilter
from src.inference.pipeline import SafetyPipeline  # 仍可用于整链（需要时）

app = FastAPI(title="CN-FastGuard", version="1.1.0")

# 在线：仅二分类
pref = Prefilter(path="./artifacts/prefilter")
THETA = None  # 若你在 thresholds.json 中维护，也可读文件；此处简单保持 None 交由客户端传参或 pipeline 统一管理

class Item(BaseModel):
    text: str
    theta: float | None = None  # 允许调用时传入不同阈值

class BatchItems(BaseModel):
    texts: List[str]
    theta: float | None = None

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post("/classify_online")
def classify_online(item: Item):
    theta = item.theta if item.theta is not None else (THETA if THETA is not None else 0.25)
    p = float(pref.predict_proba([item.text])[0])
    block = p >= theta
    return {
        "decision": "block" if block else "allow",
        "p_unsafe": p,
        "theta": theta,
        "route_offline": bool(block)   # 命中/不确定 → 交给离线
    }

@app.post("/classify_online_batch")
def classify_online_batch(items: BatchItems):
    theta = items.theta if items.theta is not None else (THETA if THETA is not None else 0.25)
    probs = list(map(float, pref.predict_proba(items.texts)))
    out = []
    for t, p in zip(items.texts, probs):
        block = p >= theta
        out.append({
            "text": t,
            "decision": "block" if block else "allow",
            "p_unsafe": p,
            "theta": theta,
            "route_offline": bool(block)
        })
    return {"results": out}

# 可选：离线深分（当你需要在服务里直接跑整链）
pipe = SafetyPipeline()

@app.post("/classify_offline")
def classify_offline(item: Item):
    # 只跑多标签可以直接用 pipeline().classify，它内置二级判别
    return pipe.classify(item.text)
