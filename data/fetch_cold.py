# data/fetch_cold.py  —— 强化版：列名自适应 + 优雅降级
import os, io, pandas as pd, requests, datasets

RAW_BASE = "https://raw.githubusercontent.com/thu-coai/COLDataset/main/COLDataset"
FILES = {"train":"train.csv","dev":"dev.csv","test":"test.csv"}

TEXT_CANDIDATES = [
    "text","content","comment","comment_text","weibo_text",
    "sentence","review","raw_text","content_text"
]
FG_CANDIDATES = ["fine-grained-label","fine_grained_label","fine-grain-label","fine_grained"]  # 攻个体/群体等
LABEL_CANDIDATES = ["label","labels","y","target"]

def _http_get(url, timeout=45):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def _pick_first(df, cands):
    for c in cands:
        if c in df.columns: 
            return c
    # 兜底：挑一个“像文本”的列（object dtype 且平均长度最大）
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        return max(lens, key=lens.get)
    return None

def _map_row(row, label_col, fine_col):
    # README: label=0 安全, 1 冒犯；test 里 fine-grained 2=attack group
    # https://github.com/thu-coai/COLDataset
    lab = int(row.get(label_col, 0))
    if lab == 0:
        return "SAFE"
    if fine_col is not None:
        try:
            fine = int(row.get(fine_col, -1))
            if fine == 2:   # 攻击群体
                return "S10"
        except Exception:
            pass
    return "UNSAFE"

def load_cold(root="./data/raw/cold"):
    os.makedirs(root, exist_ok=True)
    frames=[]
    for split, fname in FILES.items():
        local = os.path.join(root, fname)
        if not os.path.exists(local):
            try:
                buf = _http_get(f"{RAW_BASE}/{fname}")
                with open(local, "wb") as f: f.write(buf)
            except Exception:
                # 单个 split 拉取失败 → 跳过
                continue

        # 注意：有仓库会带 UTF-8 BOM
        df = pd.read_csv(local, encoding="utf-8-sig")
        # 自适应列名
        text_col  = _pick_first(df, TEXT_CANDIDATES)
        label_col = _pick_first(df, LABEL_CANDIDATES)
        fine_col  = _pick_first(df, FG_CANDIDATES)

        if label_col is None:
            # 没有标签列就没法参与映射 → 跳过该 split
            continue
        if text_col is None:
            # 仍然没法确定文本列 → 跳过该 split
            continue

        df["labels"] = df.apply(lambda r: _map_row(r, label_col, fine_col), axis=1)
        df["text_in"], df["text_out"], df["task"] = df[text_col].astype(str), "", "single"
        frames.append(df[["text_in","text_out","labels","task"]])

    if not frames:
        # 返回一个空 datasets（让上层安全拼接并继续构建）
        return datasets.Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})

    all_df = pd.concat(frames, ignore_index=True)
    return datasets.Dataset.from_pandas(all_df, preserve_index=False)
