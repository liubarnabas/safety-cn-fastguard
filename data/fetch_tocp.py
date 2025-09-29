# data/fetch_tocp.py
# -*- coding: utf-8 -*-
"""
TOCP（Chinese Profanity Processing，TRAC-2@LREC 2020）
官网提供 zip（JSON 列表），字段 original_sentence / profane_expression。
策略：若 profane_expression 非空 -> UNSAFE；否则 SAFE。
支持本地缓存目录：env TOCP_CACHE_DIR（默认 ./data/cache/tocp）
"""

from __future__ import annotations
import os, json, io, zipfile, urllib.request
from typing import List, Dict, Any
from datasets import Dataset

TOCP_ZIP_URL = "https://nlp.cse.ntou.edu.tw/resources/TOCP/TOCP_v1.0.zip"

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _ensure_zip_local() -> str | None:
    cache_dir = os.getenv("TOCP_CACHE_DIR", "./data/cache/tocp")
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "TOCP_v1.0.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return zip_path
    try:
        urllib.request.urlretrieve(TOCP_ZIP_URL, zip_path)
        return zip_path
    except Exception:
        return None

def _read_any_json_from_zip(zip_path: str) -> List[Dict[str, Any]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        # 尝试找到首个 .json 文件
        names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not names:
            return []
        for name in names:
            try:
                with zf.open(name) as f:
                    by = f.read()
                    # 支持 JSON 列表或换行 JSON
                    try:
                        data = json.loads(by.decode("utf-8"))
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                            return data["data"]
                    except Exception:
                        # 逐行 JSON
                        out = []
                        for line in io.BytesIO(by).read().decode("utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                out.append(json.loads(line))
                            except Exception:
                                continue
                        if out:
                            return out
            except Exception:
                continue
    return []

def load_tocp() -> Dataset:
    zip_path = _ensure_zip_local()
    if not zip_path:
        return _empty()
    rows = _read_any_json_from_zip(zip_path)
    if not rows:
        return _empty()

    def pick_text(o: Dict[str, Any]) -> str:
        for k in ["original_sentence", "text", "sentence"]:
            if k in o and o[k]:
                return str(o[k])
        return ""

    def is_uns(o: Dict[str, Any]) -> bool:
        pe = o.get("profane_expression")
        try:
            return bool(pe) and len(pe) > 0
        except Exception:
            return True

    outs = []
    for r in rows:
        t = pick_text(r).strip()
        if not t:
            continue
        outs.append({
            "text_in": t,
            "text_out": "",
            "labels": "UNSAFE" if is_uns(r) else "SAFE",
            "task": "single",
        })
    return Dataset.from_list(outs) if outs else _empty()
