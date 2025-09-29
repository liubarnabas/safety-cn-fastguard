# data/fetch_cdna.py
# -*- coding: utf-8 -*-
"""
Chinese Do-Not-Answer (CDNA) from Libr-AI/do-not-answer
- 自动列目录：GitHub Contents API 列出 cdna/ 下所有可下载文件
- 多格式：csv/tsv/json/jsonl（自动识别分隔符/字段名）
- 本地优先：CDNA_LOCAL 可指向本地目录或单文件
- 限流友好：可用 GITHUB_TOKEN 以提高 API 配额
- 调试：DEBUG_FETCH=1 或 --debug 打印详细日志

标签二值化（SAFE/UNSAFE）规则（尽量保守，避免误伤）：
1) 若显式“集合/性质”字段为 FP（false positive/harmless） -> SAFE
2) 若显式风控字段表明 risky/unsafe/do_not_answer=True -> UNSAFE
3) 若集合字段为 direct/general/indirect/FN（false negative） -> UNSAFE
4) 若 label in {1,true,offensive,toxic,harmful} -> UNSAFE；{0,false,harmless,safe} -> SAFE
5) 以上都推不出时，默认 SAFE（宁缺毋滥）
"""

from __future__ import annotations
import os, sys, io, csv, json, urllib.request
from typing import List, Optional, Any
from datasets import Dataset, concatenate_datasets

DEBUG = os.getenv("DEBUG_FETCH", "0") == "1"
def _dprint(*a):
    if DEBUG: print("[cdna]", *a, file=sys.stderr)

ORG  = "Libr-AI"
REPO = "do-not-answer"
DIR  = "cdna"
API  = f"https://api.github.com/repos/{ORG}/{REPO}/contents/{DIR}"
RAW  = f"https://raw.githubusercontent.com/{ORG}/{REPO}/main/{DIR}/{{name}}"
EXT_OK = (".csv", ".tsv", ".jsonl", ".json")

TEXT_CANDS  = ["question","prompt","text","instruction","query","content","title","chinese_text","zh"]
SET_CANDS   = ["set","subset","category","attack_type","question_type","type"]
RISK_CANDS  = ["is_risky","risky","unsafe","do_not_answer","dna"]
LABEL_CANDS = ["label","is_offensive","offensive","toxic","is_toxic","binary","class"]

def _empty() -> Dataset:
    return Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _read_url(url: str, headers: Optional[dict]=None) -> Optional[bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read()
    except Exception as e:
        _dprint("open fail:", url, e)
        return None

def _gh_headers() -> dict:
    h = {"User-Agent": "cdna-fetcher"}
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def _list_remote_files() -> List[str]:
    raw = _read_url(API, headers=_gh_headers())
    names: List[str] = []
    if not raw:
        _dprint("list API failed; fallback to guesses")
        return names
    try:
        arr = json.loads(raw.decode("utf-8","replace"))
        for obj in arr:
            if obj.get("type") == "file":
                name = obj.get("name","")
                if any(name.lower().endswith(ext) for ext in EXT_OK):
                    names.append(name)
        # 简单排序：先 base/overall，再 FP/FN/Direct/Indirect
        def rank(n: str) -> int:
            s = n.lower()
            if "base" in s or "all" in s: return 0
            if "fp" in s: return 1
            if "fn" in s: return 2
            if "direct" in s: return 3
            if "indirect" in s: return 4
            if "general" in s: return 5
            return 9
        names.sort(key=rank)
        if DEBUG: _dprint("remote files:", names)
    except Exception as e:
        _dprint("parse list fail:", e)
    return names

def _best_decode(raw: bytes) -> str:
    for enc in ("utf-8","utf-8-sig","gb18030","latin-1"):
        try: return raw.decode(enc, errors="replace")
        except Exception: continue
    return raw.decode("utf-8", errors="replace")

def _find_col(cols: List[str], keys: List[str]) -> Optional[str]:
    low = [c.lower() for c in cols]
    for k in keys:
        k = k.lower()
        # 既允许完全相等，也允许包含（如 "question_zh" 命中 "question"）
        for i, name in enumerate(low):
            if k == name or k in name:
                return cols[i]
    return None

def _to_bin(value: Any, set_val: str|None, risk_val: Any) -> Optional[str]:
    """按规则映射 SAFE/UNSAFE；返回 None 表示无法判断。"""
    s_set = (set_val or "").strip().lower() if set_val else ""
    if any(x in s_set for x in ["fp","false positive","harmless"]): return "SAFE"
    if any(x in s_set for x in ["fn","false negative","direct","indirect","general"]): return "UNSAFE"

    s_risk = str(risk_val).strip().lower()
    if s_risk in {"1","true","yes"}: return "UNSAFE"
    if s_risk in {"0","false","no"}: return "SAFE"

    s = str(value).strip().lower()
    if s in {"1","true","yes","offensive","toxic","harmful","unsafe"}: return "UNSAFE"
    if s in {"0","false","no","benign","harmless","safe"}: return "SAFE"

    # 某些文件可能没显式标签；无法判断时返回 None，由上层决定是否丢弃
    return None

def _rows_from_json_obj(obj: dict) -> Optional[dict]:
    cols = list(obj.keys())
    tcol = _find_col(cols, TEXT_CANDS)
    scol = _find_col(cols, SET_CANDS)
    rcol = _find_col(cols, RISK_CANDS)
    lcol = _find_col(cols, LABEL_CANDS)
    txt = (str(obj.get(tcol) or "").strip()) if tcol else ""
    if not txt: return None
    lab = _to_bin(obj.get(lcol), obj.get(scol), obj.get(rcol))
    if lab is None:
        # 没法可靠判断时，**默认 SAFE**（保守）
        lab = "SAFE"
    return {"text_in": txt, "text_out": "", "labels": lab, "task": "single"}

def _parse_text(txt: str, ext: str) -> List[dict]:
    out: List[dict] = []
    # jsonl
    if ext.endswith(".jsonl") or (txt and txt.lstrip().startswith("{") and "\n" in txt):
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            r = _rows_from_json_obj(obj)
            if r: out.append(r)
        return out

    # json
    if ext.endswith(".json"):
        try:
            j = json.loads(txt)
            if isinstance(j, list):
                for obj in j:
                    if isinstance(obj, dict):
                        r = _rows_from_json_obj(obj)
                        if r: out.append(r)
        except Exception:
            pass
        return out

    # csv/tsv
    if ext.endswith(".csv") or ext.endswith(".tsv"):
        # 简易 sniff
        delim = "," if ext.endswith(".csv") else "\t"
        try:
            sample = "\n".join(txt.splitlines()[:5])
            delim = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            pass
        reader = csv.DictReader(io.StringIO(txt), delimiter=delim)
        cols = reader.fieldnames or []
        tcol = _find_col(cols, TEXT_CANDS)
        scol = _find_col(cols, SET_CANDS)
        rcol = _find_col(cols, RISK_CANDS)
        lcol = _find_col(cols, LABEL_CANDS)
        if DEBUG:
            _dprint("csv/tsv columns:", cols, "tcol=", tcol, "scol=", scol, "rcol=", rcol, "lcol=", lcol)
        for row in reader:
            txtv = (str(row.get(tcol) or "").strip()) if tcol else ""
            if not txtv: continue
            lab = _to_bin(row.get(lcol), row.get(scol), row.get(rcol))
            if lab is None: lab = "SAFE"
            out.append({"text_in": txtv, "text_out": "", "labels": lab, "task": "single"})
        return out

    return out

def _load_remote_one(name: str) -> Optional[Dataset]:
    url = RAW.format(name=name)
    raw = _read_url(url)
    if not raw:
        return None
    txt = _best_decode(raw)
    rows = _parse_text(txt, ext=name.lower())
    if rows:
        if DEBUG: _dprint("loaded:", name, "rows=", len(rows))
        return Dataset.from_list(rows)
    else:
        if DEBUG: _dprint("no rows parsed from:", name)
        return None

def _load_from_local(path: str) -> Optional[Dataset]:
    paths: List[str] = []
    if os.path.isdir(path):
        for n in os.listdir(path):
            if any(n.lower().endswith(ext) for ext in EXT_OK):
                paths.append(os.path.join(path, n))
    elif os.path.isfile(path):
        paths.append(path)
    else:
        return None

    parts: List[Dataset] = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                raw = f.read()
            txt = _best_decode(raw)
            rows = _parse_text(txt, ext=os.path.basename(p).lower())
            if rows:
                if DEBUG: _dprint("local loaded:", os.path.basename(p), "rows=", len(rows))
                parts.append(Dataset.from_list(rows))
            else:
                if DEBUG: _dprint("local no rows:", os.path.basename(p))
        except Exception as e:
            _dprint("local read fail:", p, e)
    if parts:
        return concatenate_datasets(parts)
    return None

def load_cdna() -> Dataset:
    # 0) 本地优先
    local = os.getenv("CDNA_LOCAL")
    if local:
        d = _load_from_local(local)
        if d is not None and len(d) > 0:
            return d

    # 1) 远程列目录 + 逐个读取
    names = _list_remote_files()
    parts: List[Dataset] = []
    for name in names:
        d = _load_remote_one(name)
        if d is not None and len(d) > 0:
            parts.append(d)
    if parts:
        return concatenate_datasets(parts)

    _dprint("all candidates failed; return empty")
    return _empty()

if __name__ == "__main__":
    if "--debug" in sys.argv:
        os.environ["DEBUG_FETCH"] = "1"
    ds = load_cdna()
    print("rows:", len(ds))
    if len(ds) > 0:
        print("sample:", ds[:3])
