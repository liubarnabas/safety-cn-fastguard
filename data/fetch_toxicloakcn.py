# data/fetch_toxicloakcn.py
# -*- coding: utf-8 -*-
"""
ToxiCloakCN（EMNLP 2024）
- 自动列目录：调用 GitHub Contents API 列出 Datasets/ 下的所有文件
- 多格式支持：csv/tsv/json/jsonl（自动识别分隔符/列名）
- 中文友好：直接读 raw，UTF-8 decode（若失败回退 gb18030 / latin-1）
- 可指定本地文件/目录优先：TOXICLOAKCN_LOCAL（文件或目录均可）
- 调试：DEBUG_FETCH=1 或 --debug 打印每一步
"""

from __future__ import annotations
import os, sys, io, csv, json, re, urllib.request
from typing import List, Optional, Any
from datasets import Dataset, concatenate_datasets

DEBUG = os.getenv("DEBUG_FETCH", "0") == "1"
def _dprint(*a):
    if DEBUG: print("[toxicloakcn]", *a, file=sys.stderr)

ORG   = "Social-AI-Studio"
REPO  = "ToxiCloakCN"
DIR   = "Datasets"
API   = f"https://api.github.com/repos/{ORG}/{REPO}/contents/{DIR}"
RAW_T = f"https://raw.githubusercontent.com/{ORG}/{REPO}/main/{DIR}/{{fname}}"

# 允许的扩展名（按优先级）
EXT_OK = (".csv", ".tsv", ".jsonl", ".json")

TEXT_KEYS  = ["perturbed_text", "text", "sentence", "original_text", "content", "chinese_text", "chinese"]
LABEL_KEYS = ["label", "is_offensive", "offensive", "class", "binary", "is_toxic", "toxic"]

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _find_col(cols: List[str], keys: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for k in keys:
        k = k.lower()
        for i, name in enumerate(low):
            if k == name or k in name:
                return cols[i]
    return None

def _to_bin(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in {"1","true","yes","offensive","toxic","hate","hateful"}: return "UNSAFE"
    if s in {"0","false","no","non-offensive","benign","safe","harmless"}: return "SAFE"
    if any(k in s for k in ["offen","toxic","hate","hateful"]): return "UNSAFE"
    if any(k in s for k in ["non-offen","benign","harmless","safe"]): return "SAFE"
    return None  # 不猜测

def _read_url(url: str) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:
        _dprint("open fail:", url, e)
        return None

def _best_decode(raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return raw.decode(enc, errors="replace")
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")

def _parse_rows_from_text(txt: str, ext: str) -> List[dict]:
    rows: List[dict] = []

    # 1) jsonl
    if ext.endswith(".jsonl") or (("\n" in txt) and txt.lstrip().startswith("{")):
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cols = list(obj.keys())
            tcol = _find_col(cols, TEXT_KEYS)
            lcol = _find_col(cols, LABEL_KEYS)
            t = str(obj.get(tcol, "")).strip() if tcol else ""
            if not t: continue
            lab = _to_bin(obj.get(lcol,"")) if lcol else None
            if lab is None: continue
            rows.append({"text_in": t, "text_out": "", "labels": lab, "task": "single"})
        if rows: return rows

    # 2) json
    if ext.endswith(".json"):
        try:
            j = json.loads(txt)
            if isinstance(j, list):
                for obj in j:
                    if not isinstance(obj, dict): continue
                    cols = list(obj.keys())
                    tcol = _find_col(cols, TEXT_KEYS)
                    lcol = _find_col(cols, LABEL_KEYS)
                    t = str(obj.get(tcol, "")).strip() if tcol else ""
                    if not t: continue
                    lab = _to_bin(obj.get(lcol,"")) if lcol else None
                    if lab is None: continue
                    rows.append({"text_in": t, "text_out": "", "labels": lab, "task": "single"})
                if rows: return rows
        except Exception:
            pass

    # 3) csv/tsv（自动识别分隔符）
    if ext.endswith(".csv") or ext.endswith(".tsv"):
        # 粗略 sniff
        delim = "," if ext.endswith(".csv") else "\t"
        try:
            sniffer = csv.Sniffer()
            sample = "\n".join(txt.splitlines()[:5])
            delim = sniffer.sniff(sample).delimiter
        except Exception:
            pass
        reader = csv.DictReader(io.StringIO(txt), delimiter=delim)
        for r in reader:
            cols = list(r.keys())
            tcol = _find_col(cols, TEXT_KEYS)
            lcol = _find_col(cols, LABEL_KEYS)
            t = str(r.get(tcol, "")).strip() if tcol else ""
            if not t: continue
            lab = _to_bin(r.get(lcol,"")) if lcol else None
            if lab is None: continue
            rows.append({"text_in": t, "text_out": "", "labels": lab, "task": "single"})
        if rows: return rows

    return rows

def _list_remote_files() -> List[str]:
    # 用 GitHub Contents API 枚举 Datasets/ 下文件
    raw = _read_url(API)
    if not raw:
        _dprint("list API failed; fallback to name guesses")
        # 万一 API 被限流，给几个常见名字兜底
        return ["base.csv","homophone.csv","emoji.csv","ToxiCloakCN_base.csv","ToxiCloakCN_homophone.csv","ToxiCloakCN_emoji.csv"]
    try:
        arr = json.loads(raw.decode("utf-8","replace"))
        names = []
        for obj in arr:
            if obj.get("type") == "file":
                name = obj.get("name","")
                if any(name.lower().endswith(ext) for ext in EXT_OK):
                    names.append(name)
        # 优先按文件名包含顺序（base -> homophone -> emoji）
        def rank(n: str) -> int:
            s = n.lower()
            if "base" in s: return 0
            if "homo" in s: return 1
            if "emoji" in s: return 2
            return 3
        names.sort(key=rank)
        if DEBUG: _dprint("remote files:", names)
        return names
    except Exception as e:
        _dprint("parse list fail:", e)
        return []

def _load_one_remote(fname: str) -> Optional[Dataset]:
    url = RAW_T.format(fname=fname)
    raw = _read_url(url)
    if not raw: return None
    txt = _best_decode(raw)
    rows = _parse_rows_from_text(txt, ext=os.path.splitext(fname)[1].lower())
    if rows:
        if DEBUG: _dprint("loaded:", fname, "rows=", len(rows))
        return Dataset.from_list(rows)
    else:
        if DEBUG: _dprint("no rows parsed from:", fname)
        return None

def _load_from_local(path: str) -> Optional[Dataset]:
    # 既支持“目录”（遍历）也支持“单文件”
    files: List[str] = []
    if os.path.isdir(path):
        for n in os.listdir(path):
            if any(n.lower().endswith(ext) for ext in EXT_OK):
                files.append(os.path.join(path, n))
    elif os.path.isfile(path):
        files.append(path)
    else:
        return None

    parts: List[Dataset] = []
    for p in files:
        try:
            with open(p, "rb") as f:
                raw = f.read()
            txt = _best_decode(raw)
            rows = _parse_rows_from_text(txt, ext=os.path.splitext(p)[1].lower())
            if rows:
                if DEBUG: _dprint("local loaded:", os.path.basename(p), "rows=", len(rows))
                parts.append(Dataset.from_list(rows))
            else:
                if DEBUG: _dprint("local no rows:", os.path.basename(p))
        except Exception as e:
            _dprint("local read fail:", p, e)
            continue
    if parts:
        return concatenate_datasets(parts)
    return None

def load_toxicloakcn() -> Dataset:
    # 0) 本地优先
    local = os.getenv("TOXICLOAKCN_LOCAL")
    if local:
        d = _load_from_local(local)
        if d is not None and len(d) > 0:
            return d

    # 1) 远程遍历
    names = _list_remote_files()
    parts: List[Dataset] = []
    for fname in names:
        d = _load_one_remote(fname)
        if d is not None and len(d) > 0:
            parts.append(d)
    if parts:
        return concatenate_datasets(parts)

    # 2) 兜底：常见文件名直接拼 raw URL 尝试
    for fname in ["base.csv","homophone.csv","emoji.csv","ToxiCloakCN_base.csv","ToxiCloakCN_homophone.csv","ToxiCloakCN_emoji.csv"]:
        d = _load_one_remote(fname)
        if d is not None and len(d) > 0:
            parts.append(d)
    if parts:
        return concatenate_datasets(parts)

    _dprint("all candidates failed")
    return _empty()

if __name__ == "__main__":
    if "--debug" in sys.argv:
        os.environ["DEBUG_FETCH"] = "1"
    ds = load_toxicloakcn()
    print("rows:", len(ds))
    if len(ds) > 0:
        print("sample:", ds[:3])
