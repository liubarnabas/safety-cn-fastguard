# data/fetch_cyberbully_weibo.py
# -*- coding: utf-8 -*-
"""
Chinese Cyberbullying Post-Role（Weibo）
GitHub: kimpink98/Cyberbullying-Post-Role-Dataset
- 自动多编码解码（优先中文可读性高的编码）
- 兼容列名：'Clean Text' / 'participant role' / 'Particiant Role'
- 角色到二分类映射：{bullies, bully, aggressor, harasser, assistants, accomplice} -> UNSAFE
"""

from __future__ import annotations
from typing import Optional, List
import os, sys, io, csv, re, urllib.request
from datasets import Dataset

DEBUG = os.getenv("DEBUG_FETCH", "0") == "1"
def _dprint(*a): 
    if DEBUG: print("[cyberbully_weibo]", *a, file=sys.stderr)

REPO_RAW = "https://raw.githubusercontent.com/kimpink98/Cyberbullying-Post-Role-Dataset/main/"
CAND_FILES = [
    "Cyberbullying%20roles%20data_original.CSV",
    "Cyberbullying%20roles%20data_original.csv",
    "Cyberbullying_roles_data_original.csv",
    "Cyberbullying_roles_data_original.CSV",
]

ENC_CANDS = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936", "big5", "cp950", "latin-1"]

UNSAFE_TOKENS = {"bully","bullies","aggressor","harasser","assistant","assistants","accomplice"}

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _han_ratio(s: str) -> float:
    # 粗略衡量“中文可读性”
    if not s: return 0.0
    han = sum(1 for ch in s if '\u4e00' <= ch <= '\u9fff')
    return han / max(1, len(s))

def _find_col(cols: List[str], keys: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for k in keys:
        k = k.lower()
        for i, name in enumerate(low):
            if k in name:
                return cols[i]
    return None

def _role_to_bin(role: str) -> str:
    s = (role or "").strip().lower()
    for tok in re.split(r"[,\s/;|]+", s):
        if tok in UNSAFE_TOKENS:
            return "UNSAFE"
    return "SAFE"

def _read_raw(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:
        _dprint("open failed:", e); 
        return None

def _decode_best(raw: bytes) -> tuple[str, str]:
    """返回 (文本, 选中的编码)；用中文字符占比挑选最佳编码。"""
    best_txt, best_enc, best_score = "", "utf-8", -1.0
    for enc in ENC_CANDS:
        try:
            txt = raw.decode(enc, errors="replace")
            # 取前若干行估算中文占比
            head = "\n".join(txt.splitlines()[:100])
            score = _han_ratio(head)
            if score > best_score:
                best_txt, best_enc, best_score = txt, enc, score
        except Exception:
            continue
    if DEBUG:
        _dprint(f"selected encoding = {best_enc}, han_ratio = {best_score:.3f}")
    return best_txt, best_enc

def _parse_csv(text: str) -> Dataset | None:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return None
    cols = list(rows[0].keys())
    # 文本列兼容
    tcol = _find_col(cols, ["clean text","clean_text","text","content"])
    # 角色列兼容（含错拼 'Particiant Role'）
    rcol = _find_col(cols, ["participant role","particiant role","role"])
    if DEBUG:
        _dprint("columns:", cols, "tcol=", tcol, "rcol=", rcol)
    if not tcol:
        return None
    outs = []
    for r in rows:
        t = (r.get(tcol) or "").strip()
        if not t:
            continue
        lab = _role_to_bin(r.get(rcol, "")) if rcol else "SAFE"
        outs.append({"text_in": t, "text_out": "", "labels": lab, "task": "single"})
    return Dataset.from_list(outs) if outs else None

def load_cyberbully_weibo() -> Dataset:
    for fn in CAND_FILES:
        url = REPO_RAW + fn
        _dprint("try:", url)
        raw = _read_raw(url)
        if not raw:
            continue
        txt, enc = _decode_best(raw)
        ds = _parse_csv(txt)
        if ds is not None and len(ds) > 0:
            if DEBUG:
                _dprint("loaded rows:", len(ds))
            return ds
    _dprint("all candidates failed; return empty")
    return _empty()

if __name__ == "__main__":
    if "--debug" in sys.argv:
        os.environ["DEBUG_FETCH"] = "1"
    ds = load_cyberbully_weibo()
    print("rows:", len(ds))
    if len(ds) > 0:
        print("sample:", ds[:3])
