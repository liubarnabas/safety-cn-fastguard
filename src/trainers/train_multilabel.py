#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import List

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
)

# 让脚本可直接运行（src 布局）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.multilabel_mdeberta import MultiLabelMDeberta


LABELS: List[str] = [f"S{i}" for i in range(1, 14)]  # S1..S13


def encode_labels(labstr: str) -> np.ndarray:
    if labstr == "SAFE":
        return np.zeros(len(LABELS), dtype="float32")
    labs = set(str(labstr).split())
    y = np.zeros(len(LABELS), dtype="float32")
    for i, s in enumerate(LABELS):
        if s in labs:
            y[i] = 1.0
    return y


def build_train(ds_path: str = "./data/processed/mix_zh"):
    dsd = load_from_disk(ds_path)
    ds = dsd["train"]

    ds = ds.map(lambda e: {"y": encode_labels(e["labels"])})
    tok = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")

    keep_cols = ["text_in", "y"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]

    ds = ds.map(lambda b: tok(b["text_in"], truncation=True, max_length=384),
                batched=True, remove_columns=remove_cols)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "y"])
    return ds, tok


def main(out: str = "./artifacts/multilabel"):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ds, tok = build_train()
    model = MultiLabelMDeberta()

    def collate(batch):
        # DataCollatorWithPadding 动态补齐后，再合并我们自定义的 y
        base = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
        padded = base([{k: v for k, v in ex.items() if k in ("input_ids", "attention_mask")} for ex in batch])
        y = torch.stack([ex["y"] for ex in batch])
        padded["labels"] = y
        return padded

    args = TrainingArguments(
        out,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        dataloader_num_workers=2,
        logging_steps=50,
        save_total_limit=2
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    trainer.save_model(out)


if __name__ == "__main__":
    main()
