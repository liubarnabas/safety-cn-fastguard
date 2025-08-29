# data/fetch_toxicn.py  —— 列名自适应 + 优雅降级
import os
import pandas as pd
import requests
import datasets

URL = "https://raw.githubusercontent.com/DUT-lujunyu/ToxiCN/main/ToxiCN_1.0.csv"

# 候选正文列名（社区常见别名）
TEXT_CANDIDATES = [
    "text", "content", "comment", "comment_text",
    "weibo_text", "sentence", "raw_text", "content_text"
]

# 官方 README 稳定字段（标签）: toxic, toxic_type, expression, target
# 见: https://github.com/DUT-lujunyu/ToxiCN
LABEL_CANDS = ["toxic", "toxic_type", "expression", "target"]

def _http_get(url, timeout=45):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def _pick_text_col(df: pd.DataFrame):
    # 1) 先在候选名中找
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    # 2) 兜底：在 object 类列里选“平均长度最大”的那一列作为正文
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        return max(lens, key=lens.get)
    return None

def _map_labels(row, toxic_col="toxic", type_col="toxic_type"):
    """
    README 给出的定义：
      toxic: 0=非毒 1=毒
      toxic_type: 0=非毒, 1=一般冒犯, 2=仇恨
    我们的映射：
      toxic==0     → SAFE
      toxic==1 且 toxic_type==2 → S10 (Hate)
      其他毒性     → UNSAFE （交由主模型细化）
    """
    try:
        tox = int(row.get(toxic_col, 0))
    except Exception:
        tox = 0
    if tox == 0:
        return "SAFE"
    try:
        ttype = int(row.get(type_col, 0))
    except Exception:
        ttype = 0
    if ttype == 2:
        return "S10"
    return "UNSAFE"

def load_toxicn(path="./data/raw/toxicn/ToxiCN_1.0.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 若本地无文件，尝试下载；失败则优雅返回空数据集
    if not os.path.exists(path):
        try:
            buf = _http_get(URL)
            with open(path, "wb") as f:
                f.write(buf)
        except Exception:
            return datasets.Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

    # 注意 UTF-8 BOM
    try_enc = ["utf-8-sig", "utf-8"]
    for enc in try_enc:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return datasets.Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

    # 自动探测正文列
    text_col = _pick_text_col(df)
    if text_col is None:
        return datasets.Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

    # 确认标签列存在（至少 toxic / toxic_type 能取到）
    if "toxic" not in df.columns:
        # 极端副本：没有 toxic 列，无法映射 → 直接跳过
        return datasets.Dataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

    # 生成标签
    df["labels"] = df.apply(lambda r: _map_labels(r, toxic_col="toxic", type_col="toxic_type" if "toxic_type" in df.columns else None), axis=1)
    df["text_in"] = df[text_col].astype(str)
    df["text_out"], df["task"] = "", "single"
    return datasets.Dataset.from_pandas(df[["text_in", "text_out", "labels", "task"]], preserve_index=False)
