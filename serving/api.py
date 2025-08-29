#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# 让脚本可直接运行（src 布局）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.inference.pipeline import SafetyPipeline

app = FastAPI(title="CN-FastGuard", version="1.0.0")
pipe = SafetyPipeline()  # 读取 src/inference/thresholds.json

class Item(BaseModel):
    text: str

class BatchItems(BaseModel):
    texts: List[str]

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/classify")
def classify(item: Item):
    return pipe.classify(item.text)

@app.post("/classify_batch")
def classify_batch(items: BatchItems):
    return pipe.classify_batch(items.texts)
