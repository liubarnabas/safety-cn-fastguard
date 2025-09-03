# data/build_binary_zh.py
# -*- coding: utf-8 -*-
"""
将多源中文安全数据集合并为【前置二分类】训练集（SAFE / UNSAFE）：
- PolyGuardMix（中文子集）
- COLD
- ToxiCN
- SWSR
- ChineseSafe（多候选 HF 仓已在 fetch_chinesesafe 中处理）
- 本地 raw/xiaopen.csv

功能要点：
1) 统一字段：text_in / text_out / labels / task
2) 标签二值化：除 "SAFE" 外一律并为 "UNSAFE"
3) 规范化文本 + 指纹去重（更强的“近重复”压制）
4) 可选下采样 SAFE 以平衡正负样本
5) 保存至 ./data/processed/binary_zh
"""

import argparse
from typing import List, Tuple
import re, unicodedata, hashlib

from datasets import Dataset, DatasetDict, concatenate_datasets

from fetch_polyguardmix_zh import load_polyguardmix_zh
from fetch_cold import load_cold
from fetch_toxicn import load_toxicn
from fetch_swsr import load_swsr
from fetch_chinesesafe import load_chinesesafe
from fetch_xiaopen import load_xiaopen


# ------------------------------
# 文本规范化（提高去重质量与训练稳定性）
# ------------------------------

_ZWSP_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_PUNC_RE = re.compile(r"([，。、“”‘’！!？?\.\,])\1{1,}")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def normalize_zh(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = _ZWSP_RE.sub("", t)
    t = _URL_RE.sub("<URL>", t)
    t = _EMOJI_RE.sub("<EMOJI>", t)
    t = _MULTI_PUNC_RE.sub(r"\1", t)
    t = _WS_RE.sub(" ", t).strip()
    return t

def fingerprint(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ------------------------------
# 工具函数
# ------------------------------

def _empty_dataset() -> Dataset:
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({"text_in": [], "text_out": [], "labels": [], "task": []})

def _norm_label_to_binary(label: str) -> str:
    """只要不是严格等于 'SAFE'（大小写无关），就并为 'UNSAFE'。"""
    if label is None:
        return "UNSAFE"
    if str(label).strip().upper() == "SAFE":
        return "SAFE"
    return "UNSAFE"

def _to_binary(ds: Dataset) -> Dataset:
    """将任意来源数据集统一为二分类格式，并清理多余列（加入规范化与指纹字段，用于强去重）。"""
    if ds is None or len(ds) == 0:
        return _empty_dataset()

    keep = {"text_in", "text_out", "labels", "task", "text_norm", "fp"}

    def _mapper(e):
        lab = _norm_label_to_binary((e.get("labels") or ""))
        raw = str(e.get("text_in") or "").strip()
        norm = normalize_zh(raw)
        return {
            "text_in": norm,     # 直接使用规范化文本训练
            "text_out": "",
            "labels": lab,
            "task": "single",
            "text_norm": norm,   # 辅助列：规范化文本
            "fp": fingerprint(norm)  # 辅助列：指纹
        }

    ds2 = ds.map(_mapper)
    ds2 = ds2.filter(lambda e: bool(e["text_in"]))
    rm_cols = [c for c in ds2.column_names if c not in keep]
    if rm_cols:
        ds2 = ds2.remove_columns(rm_cols)
    return ds2

def _dedup_by_text(ds: Dataset) -> Dataset:
    """按规范化后指纹去重（保持首个出现样本），最后移除辅助列。"""
    seen = set()
    def _keep(e):
        f = e["fp"]
        if f in seen:
            return False
        seen.add(f)
        return True
    ds = ds.filter(_keep)
    return ds.remove_columns([c for c in ["text_norm", "fp"] if c in ds.column_names])

def _concat_nonempty(parts: List[Dataset]) -> Dataset:
    parts = [d for d in parts if d is not None and len(d) > 0]
    if not parts:
        return _empty_dataset()
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)

def _balance_safe(ds_all: Dataset, ratio: float, seed: int = 42) -> Dataset:
    """
    下采样 SAFE：保留 SAFE 数量 ≈ ratio * UNSAFE 数量。
    ratio=1.0 表示 SAFE 与 UNSAFE 大致均衡；>=1.0 不采样；<=0 跳过。
    """
    if ratio <= 0 or ratio >= 1.0:
        return ds_all

    ds_safe = ds_all.filter(lambda e: e["labels"] == "SAFE")
    ds_uns = ds_all.filter(lambda e: e["labels"] == "UNSAFE")
    if len(ds_safe) == 0 or len(ds_uns) == 0:
        return ds_all

    n_keep = int(len(ds_uns) * ratio)
    if n_keep <= 0:
        return ds_all
    n_keep = min(n_keep, len(ds_safe))
    ds_safe = ds_safe.shuffle(seed=seed).select(range(n_keep))
    return concatenate_datasets([ds_safe, ds_uns]).shuffle(seed=seed)

def _source_stats(sources: List[Tuple[str, Dataset]]) -> str:
    lines = []
    for name, ds in sources:
        n = len(ds) if ds is not None else 0
        lines.append(f"- {name:<16}: {n:>7d}")
    return "\n".join(lines)

def _binary_stats(ds: Dataset) -> str:
    if ds is None or len(ds) == 0:
        return "Total: 0"
    n_all = len(ds)
    n_safe = len(ds.filter(lambda e: e["labels"] == "SAFE"))
    n_uns = len(ds.filter(lambda e: e["labels"] == "UNSAFE"))
    return f"Total: {n_all} | SAFE: {n_safe} | UNSAFE: {n_uns}"


# ------------------------------
# 主流程
# ------------------------------

def build(
    balance_safe_ratio: float = 1.0,
    shuffle_seed: int = 42,
) -> DatasetDict:
    """构建二分类训练集，返回 DatasetDict(train=...)。"""

    # 1) 载入各源（原始 -> 二分类）
    src_raw = [
        ("polyguardmix_zh", load_polyguardmix_zh()),
        ("cold",            load_cold()),
        ("toxicn",          load_toxicn()),
        ("swsr",            load_swsr()),
        ("chinesesafe",     load_chinesesafe()),
#        ("xiaopen_local",   load_xiaopen()),
    ]
    src_bin = [(name, _to_binary(ds)) for name, ds in src_raw]

    # 2) 合并 & 去重
    ds_all = _concat_nonempty([ds for _, ds in src_bin])
    ds_all = _dedup_by_text(ds_all)

    # 3) 下采样 SAFE（可选）
    ds_all = _balance_safe(ds_all, ratio=balance_safe_ratio, seed=shuffle_seed)

    # 4) 打散 & 打包
    ds_all = ds_all.shuffle(seed=shuffle_seed)
    return DatasetDict({"train": ds_all})


def main():
    parser = argparse.ArgumentParser(description="Build binary (SAFE/UNSAFE) zh dataset for online classifier.")
    parser.add_argument("--balance-safe-ratio", type=float, default=1.0,
                        help="SAFE 下采样比例：保留 SAFE ≈ ratio * (#UNSAFE)。默认 1.0（不采样）。例如 0.8 表示 SAFE 数量约为 UNSAFE 的 0.8 倍。")
    parser.add_argument("--save-path", type=str, default="./data/processed/binary_zh",
                        help="保存路径（datasets.save_to_disk 目录）。")
    parser.add_argument("--no-stats", action="store_true", help="不打印统计信息。")
    args = parser.parse_args()

    # 构建
    dsd = build(balance_safe_ratio=args.balance_safe_ratio)

    # 打印统计
    if not args.no_stats:
        # 再次加载二分类版源统计
        src_bin = [
            ("polyguardmix_zh", _to_binary(load_polyguardmix_zh())),
            ("cold",            _to_binary(load_cold())),
            ("toxicn",          _to_binary(load_toxicn())),
            ("swsr",            _to_binary(load_swsr())),
            ("chinesesafe",     _to_binary(load_chinesesafe())),
            ("xiaopen_local",   _to_binary(load_xiaopen())),
        ]
        print("[Per-source rows after binarization]")
        print(_source_stats(src_bin))
        print("\n[Final merged stats]")
        print(_binary_stats(dsd["train"]))

    # 保存
    dsd.save_to_disk(args.save_path)
    print(f"\nSaved to: {args.save_path}")


if __name__ == "__main__":
    main()
