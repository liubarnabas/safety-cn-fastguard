#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)

# 让脚本在 src/ 布局下可直接运行
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class FocalLossCE(nn.Module):
    """二分类 CrossEntropy 上叠加 Focal（应对不平衡；与 CE 一起用于精度优先场景）"""
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        # 注意：weight 的设备/精度需与 logits 一致；在 forward 里动态迁移
        self.weight = weight  # shape [2]

    def forward(self, logits, target, weight: Optional[torch.Tensor] = None):
        # logits: [B,2], target: [B]
        w = weight if weight is not None else self.weight
        if w is not None:
            w = w.to(device=logits.device, dtype=logits.dtype)
        # 直接用 functional 版本，传入对齐后的 weight
        ce = nn.functional.cross_entropy(logits, target, weight=w)
        logp = torch.log_softmax(logits, dim=-1)
        pt = torch.exp(logp.gather(1, target.view(-1, 1))).squeeze(1)  # prob of true class
        focal = (1.0 - pt).clamp_min(0).pow(self.gamma)
        return focal.mean() * ce


class TemperatureScaler(nn.Module):
    """Guo et al., 2017：温度缩放，校准概率，不改变准确率"""
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / torch.exp(self.logT)

    @torch.no_grad()
    def temperature(self):
        return float(torch.exp(self.logT).item())


def _split_train_val(ds, val_size=0.12, seed=42):
    idx = np.arange(len(ds))
    tr, va = train_test_split(idx, test_size=val_size, random_state=seed, shuffle=True)
    return ds.select(tr.tolist()), ds.select(va.tolist())


def main(ds_path="./data/processed/mix_zh",
         out="./artifacts/prefilter",
         use_focal=False, focal_gamma=1.5,
         class_weight_for_accuracy=True):
    dsd = load_from_disk(ds_path)
    ds = dsd["train"]

    # SAFE->0, others->1
    def binlab(e): return {"label": 0 if e["labels"] == "SAFE" else 1}
    ds = ds.map(binlab)

    # 留出验证集
    train_ds, val_ds = _split_train_val(ds, val_size=0.12, seed=42)

    name = "hfl/chinese-roberta-wwm-ext"
    tok = AutoTokenizer.from_pretrained(name)

    keep_cols = ["text_in", "label"]
    def tok_fn(b): return tok(b["text_in"], truncation=True, max_length=256)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep_cols])
    val_ds   = val_ds.map(tok_fn,   batched=True, remove_columns=[c for c in val_ds.column_names if c not in keep_cols])

    data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    # 轻微偏向“高准确率”（减少 FP）：unsafe 类权重设为 0.9
    class_weight = None
    if class_weight_for_accuracy:
        class_weight = torch.tensor([1.0, 0.9], dtype=torch.float32)  # 设备/精度在用时对齐

    loss_obj = FocalLossCE(gamma=focal_gamma, weight=class_weight) if use_focal else None

    class MyTrainer(Trainer):
        # 兼容 4.5x：接受 num_items_in_batch
        def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # 保证 weight 与 logits 同设备/同 dtype（AMP 下 logits 可能是 fp16）
            w = class_weight.to(device=logits.device, dtype=logits.dtype) if class_weight is not None else None
            if loss_obj is None:
                loss = nn.functional.cross_entropy(logits, labels, weight=w)
            else:
                loss = loss_obj(logits, labels, weight=w)
            return (loss, outputs) if return_outputs else loss

    # eval 与 save 策略都用 "epoch"，以支持 load_best_model_at_end
    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        dataloader_num_workers=2,
        logging_steps=50,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": float((preds == labels).mean())}

    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)

    # -------- 温度缩放：用验证集最小化 NLL 学习 T --------
    trainer.model.eval()
    scaler = TemperatureScaler().to(trainer.model.device)
    scaler.train()
    nll = nn.CrossEntropyLoss()
    optim = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=100)
    val_loader = trainer.get_eval_dataloader(val_ds)

    def closure():
        optim.zero_grad(set_to_none=True)
        batch = next(iter(val_loader))
        batch = {k: v.to(trainer.model.device) for k, v in batch.items()}
        with torch.no_grad():
            logits = trainer.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        scaled = scaler(logits)
        loss = nll(scaled, batch["labels"])
        loss.backward()
        return loss

    for _ in range(25):
        optim.step(closure)

    T = scaler.temperature()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"T": T}, f)
    print(f"[TemperatureScaling] learned T={T:.4f}")


if __name__ == "__main__":
    main()
