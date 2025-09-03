# data/fetch_xiaopen.py
import os
import pandas as pd
import datasets

TEXT_CANDIDATES = [
    "content", "text", "comment", "comment_text", "weibo_text",
    "sentence", "raw_text", "content_text", "prompt"
]
LABEL_CANDIDATES = ["label", "labels", "y", "target", "safety", "is_safe", "toxic", "toxicity"]

def _pick_text_col(df: pd.DataFrame):
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    # 兜底：选平均长度最大的 object 列
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        return max(lens, key=lens.get)
    return None

def _pick_label_col(df: pd.DataFrame):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _to_bin(v):
    # 0/1 数值优先；文本兜底
    try:
        iv = int(v)
        return "UNSAFE" if iv == 1 else "SAFE"
    except Exception:
        s = str(v).strip().lower()
        if s in {"1","unsafe","有害","违规","冒犯","仇恨","毒性","不安全"}:
            return "UNSAFE"
        return "SAFE"

def load_xiaopen(path="./data/raw/xiaopen.csv") -> datasets.Dataset:
    # 如果文件不存在，返回空数据集（不影响构建流程）
    if not os.path.exists(path):
        return datasets.Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

    # 兼容 UTF-8 BOM
    tried, df = False, None
    for enc in ("utf-8-sig","utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc)
            tried = True
            break
        except Exception:
            continue
    if not tried or df is None or df.empty:
        return datasets.Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

    text_col = _pick_text_col(df)
    label_col = _pick_label_col(df)
    if text_col is None or label_col is None:
        return datasets.Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

    df = df[[text_col, label_col]].copy()
    df["text_in"] = df[text_col].astype(str)
    df["labels"]  = df[label_col].apply(_to_bin)
    df["text_out"] = ""
    df["task"] = "single"
    return datasets.Dataset.from_pandas(df[["text_in","text_out","labels","task"]], preserve_index=False)
