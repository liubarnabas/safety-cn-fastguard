# data/fetch_chinesesafe.py
# -*- coding: utf-8 -*-
"""
只从 HuggingFace 加载 ChineseSafe（SUSTech/ChineseSafe），并映射为二分类 SAFE/UNSAFE。
可用环境变量 CHINESESAFE_SPLIT 指定 split（如 test、train 等）；若未指定，优先尝试 all，
否则依次尝试 train/validation/dev/test 并合并。
输出统一字段：text_in / text_out / labels / task
"""

from __future__ import annotations
import os
from typing import Any, Optional, List

import datasets
from datasets import Dataset, load_dataset, concatenate_datasets


def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})


def _to_bin(v: Any) -> str:
    # 数值优先：1→UNSAFE, 0→SAFE
    try:
        iv = int(v)
        return "UNSAFE" if iv == 1 else "SAFE"
    except Exception:
        pass

    s = str(v).strip().lower()
    if not s:
        return "SAFE"

    unsafe_kw = [
        "1", "unsafe", "toxic", "abusive", "offensive", "hate", "sexual", "drug", "self-harm",
        "违", "不安全", "有害", "仇恨", "歧视", "偏见", "冒犯", "辱骂", "毒", "违法",
        "暴力", "极端", "恐怖", "诈骗", "成人", "色情", "自杀", "自残"
    ]
    safe_kw = ["0", "safe", "benign", "harmless", "non-toxic", "clean", "正常", "安全", "合规", "无害"]

    if any(k in s for k in unsafe_kw):
        return "UNSAFE"
    if any(k in s for k in safe_kw):
        return "SAFE"
    # 未知文本标签：默认 SAFE，避免伪阳性污染
    return "SAFE"


def _map_ex(e: dict) -> dict:
    return {
        "text_in":  str(e.get("text") or "").strip(),
        "text_out": "",
        "labels":   _to_bin(e.get("label", "")),
        "task":     "single",
    }


def load_chinesesafe(split: Optional[str] = None) -> Dataset:
    """
    只读 HF：SUSTech/ChineseSafe
    优先使用传入 split 或环境变量 CHINESESAFE_SPLIT；
    否则尝试 all；再否则合并 train/validation/dev/test 中能成功读取的部分。
    """
    forced = split or os.getenv("CHINESESAFE_SPLIT")

    if forced:
        try:
            d = load_dataset("SUSTech/ChineseSafe", split=forced)
            return d.map(_map_ex, remove_columns=d.column_names)
        except Exception:
            return _empty()

    # 尝试 all（若存在最简单）
    try:
        d = load_dataset("SUSTech/ChineseSafe", split="all")
        return d.map(_map_ex, remove_columns=d.column_names)
    except Exception:
        pass

    # 依次尝试常见 splits 并合并
    frames: List[Dataset] = []
    for sp in ["train", "validation", "dev", "test"]:
        try:
            d = load_dataset("SUSTech/ChineseSafe", split=sp)
            frames.append(d.map(_map_ex, remove_columns=d.column_names))
        except Exception:
            continue

    if frames:
        return concatenate_datasets(frames)
    return _empty()
