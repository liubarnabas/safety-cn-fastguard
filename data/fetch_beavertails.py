# data/fetch_beavertails.py
# -*- coding: utf-8 -*-
"""
BeaverTails（HuggingFace：PKU-Alignment/BeaverTails）
修复点：
  - 自动探测 split（round0/330k/30k 等历史命名差异统一处理）
  - 中文优先；若为 0 且 ALLOW_FALLBACK_ENGLISH=1（默认）回退英文
  - --debug 打印 config/split；跳过没有 is_safe 的评测集（如 BeaverTails-Evaluation）
官方仓库 README 和数据卡指向该 HF 数据集。
"""

from __future__ import annotations
import os, re, sys
from typing import List

from datasets import Dataset, load_dataset, get_dataset_config_names, get_dataset_split_names, concatenate_datasets

DEBUG = os.getenv("DEBUG_FETCH","0")=="1"
def _dprint(*a):
    if DEBUG: print("[beavertails]", *a, file=sys.stderr)

HF_ID = "PKU-Alignment/BeaverTails"
ZH_RE = re.compile(r"[\u4e00-\u9fff]")

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

def _map_one(d: Dataset) -> Dataset:
    cols = d.column_names
    def mapper(e):
        prompt = str(e.get("prompt","") or e.get("question","") or e.get("instruction","")).strip()
        resp   = str(e.get("response","") or e.get("answer","")).strip()
        is_safe = e.get("is_safe", None)
        if is_safe is None:
            # 没有标签（可能是评测集），返回 None 以便后续过滤
            return None
        txt = prompt if prompt else resp
        lab = "SAFE" if bool(is_safe) else "UNSAFE"
        return {"text_in": txt, "text_out":"", "labels": lab, "task":"single"}
    d2 = d.map(mapper, remove_columns=cols)
    d2 = d2.filter(lambda e: e is not None and bool(e["text_in"]))
    return d2

def load_beavertails() -> Dataset:
    try:
        cfgs = get_dataset_config_names(HF_ID)
    except Exception as e:
        _dprint("get configs failed:", e); cfgs=[None]
    if not cfgs: cfgs=[None]

    parts: List[Dataset] = []
    for cfg in cfgs:
        try:
            splits = get_dataset_split_names(HF_ID, cfg) if cfg else get_dataset_split_names(HF_ID)
        except Exception as e:
            _dprint("get splits failed:", cfg, e)
            splits = ["train","test","validation","330k_train","330k_test","30k_train","30k_test"]
        loaded = False
        tried_splits = list(dict.fromkeys(["train","test","validation","330k_train","330k_test","30k_train","30k_test"] + splits))
        for sp in tried_splits:
            try:
                d = load_dataset(HF_ID, cfg, split=sp) if cfg else load_dataset(HF_ID, split=sp)
                parts.append(_map_one(d))
                loaded = True
                _dprint("loaded:", cfg, sp, "rows=", len(d))
            except Exception:
                continue
        if not loaded:
            _dprint("no split loaded for cfg:", cfg)

    if not parts:
        _dprint("no parts loaded; return empty")
        return _empty()

    d = concatenate_datasets(parts)
    d = d.filter(lambda e: e is not None and bool(e["labels"]))  # 过滤掉评测无标签样本
    d_zh = d.filter(lambda e: bool(ZH_RE.search(e["text_in"])))
    if len(d_zh)>0:
        return d_zh
    allow_en = os.getenv("ALLOW_FALLBACK_ENGLISH","1")=="1"
    _dprint("no Chinese samples; allow_en=", allow_en, "return_en_rows=", len(d))
    return d if allow_en else _empty()

if __name__ == "__main__":
    if "--debug" in sys.argv: os.environ["DEBUG_FETCH"]="1"
    ds = load_beavertails()
    print("rows:", len(ds))
    if len(ds)>0: print("sample:", ds[:3])
