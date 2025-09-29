# data/fetch_chsd.py
# -*- coding: utf-8 -*-
"""
CHSD（Chinese Hate Speech Detection）：GitHub 提供 train/test CSV。
仓库：RXJ588/CHSD，包含 17430 条标注，label: 0 安全 / 1 仇恨。
输出统一字段：text_in / text_out / labels / task
"""

from __future__ import annotations
from typing import List
from datasets import Dataset, load_dataset, concatenate_datasets

RAW_TRAIN = "https://raw.githubusercontent.com/RXJ588/CHSD/main/train.csv"
RAW_TEST  = "https://raw.githubusercontent.com/RXJ588/CHSD/main/test.csv"

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _pick_text(ex):
    # 猜测列名：常见为 text / sentence / content
    for k in ["text", "sentence", "content", "Text", "Sentence", "内容"]:
        v = ex.get(k)
        if v is not None:
            return str(v)
    # 若 CSV 单列，datasets 会命名为 "Unnamed: 0" + 一列文本；此处尽量遍历
    for k, v in ex.items():
        if k.lower() not in ("label", "labels"):
            if isinstance(v, str):
                return v
    return ""

def _map_ex(ex):
    # label: 1 -> UNSAFE, 0 -> SAFE
    lab = ex.get("label", ex.get("labels", 0))
    try:
        lab = int(lab)
    except Exception:
        lab = 1
    return {
        "text_in":  _pick_text(ex).strip(),
        "text_out": "",
        "labels":   "UNSAFE" if lab == 1 else "SAFE",
        "task":     "single",
    }

def load_chsd() -> Dataset:
    files: List[str] = [RAW_TRAIN, RAW_TEST]
    frames = []
    for url in files:
        try:
            d = load_dataset("csv", data_files=url, split="train")
            frames.append(d.map(_map_ex, remove_columns=d.column_names))
        except Exception:
            continue
    if frames:
        return concatenate_datasets(frames)
    return _empty()
