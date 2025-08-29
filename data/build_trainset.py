# data/build_trainset.py
from datasets import concatenate_datasets, DatasetDict
from fetch_polyguardmix_zh import load_polyguardmix_zh
from fetch_cold import load_cold
from fetch_toxicn import load_toxicn
from fetch_swsr import load_swsr

def build():
    ds_pg = load_polyguardmix_zh()
    # 其余三个“可能为空”；仅在len>0时并入
    parts = [ds_pg]
    for loader in (load_cold, load_toxicn, load_swsr):
        d = loader()
        if len(d) > 0: parts.append(d)
    ds_all = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return DatasetDict({"train": ds_all.shuffle(seed=42)})

if __name__ == "__main__":
    dsd = build()
    dsd.save_to_disk("./data/processed/mix_zh")
