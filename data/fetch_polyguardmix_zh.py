# data/fetch_polyguardmix_zh.py
from datasets import load_dataset, DatasetDict, Dataset
def load_polyguardmix_zh():
    ds = load_dataset("ToxicityPrompts/PolyGuardMix", split="train")  # parquet 已提供
    def is_zh(example):
        meta = example.get("metadata") or {}
        lang = meta.get("language") or meta.get("lang")
        return str(lang).lower().startswith("zh") or str(lang)=="Chinese"
    ds_zh = ds.filter(is_zh)
    # 统一字段： text_in / text_out / labels (S-classes) / task
    def map_item(e):
        s = (e.get("prompt_safety_categories") or "") + " " + (e.get("response_safety_categories") or "")
        s = " ".join(sorted({t.strip() for t in s.split() if t.strip().startswith("S")}))
        return {
            "text_in": e["prompt"],
            "text_out": e.get("response") or "",
            "labels": s.strip() or "SAFE",
            "task": "pair"  # prompt+response 场景
        }
    return ds_zh.map(map_item, remove_columns=ds.column_names)
