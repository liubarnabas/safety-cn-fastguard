# server/prefilter_service.py
# -*- coding: utf-8 -*-
"""
Safety Prefilter (Binary) FastAPI 服务
- 批量预测：/score
- 健康检查：/healthz
- 自动加载 temperature.json (T) 与 theta.json (阈值)
- 优先 ONNXRuntime（若存在 model.onnx），否则回退 PyTorch
- 兼容从任意工作目录启动（自动把项目根目录加入 sys.path）
"""

import os
import sys
from typing import List, Optional

# ---------- 将项目根目录加入 sys.path ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # server/ 的上一级
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------- 现在可以安全地 import src 下的模块 ----------
from fastapi import FastAPI
from pydantic import BaseModel
from src.models.prefilter_binary import Prefilter

APP_TITLE = "Safety Prefilter (Binary)"
APP_DESC = "SAFE/UNSAFE 二分类前置过滤器服务（支持批量、温度缩放、ONNX/CPU/GPU）。"
APP_VER = "1.0.0"

class ScoreRequest(BaseModel):
    texts: List[str]
    temperature: Optional[float] = None  # 可临时指定 T（不覆盖默认 T）

class ScoreResponse(BaseModel):
    probs: List[float]  # 属于 UNSAFE 的概率
    labels: List[int]   # 1=UNSAFE, 0=SAFE
    theta: float
    T: float

app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VER)

# 环境变量：
#   PREFILTER_PATH: 模型目录（含 tokenizer、temperature.json、theta.json、model.onnx）
#   USE_ONNX: "1/true" 则优先 ONNX
PREFILTER_PATH = os.getenv("PREFILTER_PATH", "./artifacts/prefilter_roberta")
USE_ONNX = str(os.getenv("USE_ONNX", "1")).lower() in {"1", "true", "yes"}

# 初始化模型（优先 ONNX）
pref = Prefilter(path=PREFILTER_PATH, use_onnx=USE_ONNX)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "theta": float(pref.theta), "T": float(pref.T)}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    probs = pref.predict_proba(req.texts, T=req.temperature)
    theta = float(pref.theta)
    labels = [1 if p >= theta else 0 for p in probs]
    T_eff = float(pref.T if req.temperature is None else req.temperature)
    return ScoreResponse(probs=probs, labels=labels, theta=theta, T=T_eff)

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    # 单进程即可；如需多 worker，建议用 gunicorn/uvicorn 多进程模式在容器/PM2 层面管理
    uvicorn.run("server.prefilter_service:app", host=host, port=port, workers=1, reload=False)
