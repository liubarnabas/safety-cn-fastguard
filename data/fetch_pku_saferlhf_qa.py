# data/fetch_pku_saferlhf_qa.py
# -*- coding: utf-8 -*-
"""
PKU-Alignment/PKU-SafeRLHF-QA（HuggingFace）
修复点：
  - 自动探测可用的 config 与 split（有的卡只有 train，有的也有 validation/test）
  - 中文优先；若中文为 0 且 ALLOW_FALLBACK_ENGLISH=1（默认）则回退英文
  - --debug 打印可用 config/split 与样例字段
数据卡：PKU-Alignment/PKU-SafeRLHF-QA
"""

from __future__ import annotations
import os, re, sys
from typing import List

from datasets import Dataset, load_dataset, get_dataset_config_names, get_dataset_split_names, concatenate_datasets

DEBUG = os.getenv("DEBUG_FETCH","0")=="1"
def _dprint(*a):
    if DEBUG: print("[pku_saferlhf_qa]", *a, file=sys.stderr)

HF_ID = "PKU-Alignment/PKU-SafeRLHF-QA"
ZH_RE = re.compile(r"[\u4e00-\u9fff]")

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

def _to_bin(is_safe) -> str:
    try:
        return "SAFE" if bool(is_safe) else "UNSAFE"
    except Exception:
        return "SAFE"

def _map_one(d: Dataset) -> Dataset:
    cols = d.column_names
    def mapper(e):
        prompt = str(e.get("prompt","") or e.get("question","") or e.get("instruction","")).strip()
        resp   = str(e.get("response","") or e.get("answer","")).strip()
        txt = prompt if prompt else resp
        return {"text_in": txt, "text_out":"", "labels": _to_bin(e.get("is_safe", True)), "task":"single"}
    return d.map(mapper, remove_columns=cols).filter(lambda e: bool(e["text_in"]))

def load_pku_saferlhf_qa() -> Dataset:
    # 发现 configs 与 splits
    try:
        cfgs = get_dataset_config_names(HF_ID)
    except Exception as e:
        _dprint("get configs failed:", e)
        return _empty()
    if not cfgs: cfgs = [None]

    parts: List[Dataset] = []
    for cfg in cfgs:
        try:
            splits = get_dataset_split_names(HF_ID, cfg) if cfg else get_dataset_split_names(HF_ID)
        except Exception as e:
            _dprint("split names failed for", cfg, e)
            splits = ["train","validation","test"]
        tried = False
        for sp in ["train","validation","test","all"] + list(splits):
            try:
                d = load_dataset(HF_ID, cfg, split=sp) if cfg else load_dataset(HF_ID, split=sp)
                tried = True
                parts.append(_map_one(d))
                _dprint("loaded:", cfg, sp, "rows=", len(d))
            except Exception as e:
                continue
        if not tried:
            _dprint("no split loaded for cfg:", cfg)

    if not parts:
        _dprint("no parts loaded; return empty")
        return _empty()

    d = concatenate_datasets(parts)
    d_zh = d.filter(lambda e: bool(ZH_RE.search(e["text_in"])))
    if len(d_zh)>0:
        return d_zh
    allow_en = os.getenv("ALLOW_FALLBACK_ENGLISH","1")=="1"
    _dprint("no Chinese samples; allow_en=", allow_en, "return_en_rows=", len(d))
    return d if allow_en else _empty()

if __name__ == "__main__":
    if "--debug" in sys.argv: os.environ["DEBUG_FETCH"]="1"
    ds = load_pku_saferlhf_qa()
    print("rows:", len(ds))
    if len(ds)>0: print("sample:", ds[:3])
