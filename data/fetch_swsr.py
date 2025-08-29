# data/fetch_swsr.py
import os, pandas as pd, requests, datasets
BASE = "https://raw.githubusercontent.com/aggiejiang/SWSR/main/SWSR"
FILE = "SexComment.csv"

def _http_get(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def load_swsr(path="./data/raw/swsr/SexComment.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        try:
            with open(path, "wb") as f: f.write(_http_get(f"{BASE}/{FILE}"))
        except Exception:
            return datasets.Dataset.from_dict({"text_in":[], "text_out":[], "labels":[], "task":[]})
    df = pd.read_csv(path)
    df["labels"] = df["label"].apply(lambda v: "S10" if int(v)==1 else "SAFE")
    df["text_in"], df["text_out"], df["task"] = df["comment_text"], "", "single"
    return datasets.Dataset.from_pandas(df[["text_in","text_out","labels","task"]], preserve_index=False)
