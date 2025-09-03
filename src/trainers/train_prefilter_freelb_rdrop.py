#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线二分类（Accuracy 优先）——稳健版精调配方：
FreeLB(保守超参, 可关) + R-Drop(可关/弱化) + SWA(可关) + LLRD(可关) + EarlyStopping + 温度缩放

变更要点 vs 你当前版本：
  1) FreeLB 默认大幅降噪：eps=0.3, alpha=0.1, steps=2（来自论文常见区间）
  2) R-Drop 默认 alpha=0.2，并允许完全关闭
  3) 默认关闭 fp16（对抗训练在半精度下更易数值不稳；如需可显式 --fp16 true）
  4) 统一使用 'labels' 键，避免 label/labels 混淆
  5) 加 seed、梯度裁剪、warmup_ratio 参数；LLRD/SWA 可一键关
  6) 训练日志首行打印当前所有关键超参，便于复现实验
"""
import os, sys, json, math, re, random
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup, set_seed
)

# --- src 布局 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def map_binary_label(ex: Dict[str, Any]) -> Dict[str, Any]:
    # SAFE -> 0, others -> 1
    return {"label": 0 if ex["labels"] == "SAFE" else 1}


def build_dataset(ds_path: str, backbone_name: str, max_len: int = 256):
    dsd = load_from_disk(ds_path)
    ds = dsd["train"].map(map_binary_label)

    # 留出验证集（datasets 自带）
    dss = ds.train_test_split(test_size=0.12, seed=42)
    train_ds, val_ds = dss["train"], dss["test"]

    # 统一改名为 "labels"（避免 label/labels 不一致）
    if "label" in train_ds.column_names:
        train_ds = train_ds.rename_column("label", "labels")
        val_ds   = val_ds.rename_column("label", "labels")

    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    keep_cols = ["text_in", "labels"]

    def tok_fn(b): return tok(b["text_in"], truncation=True, max_length=max_len)

    train_ds = train_ds.map(tok_fn, batched=True,
        remove_columns=[c for c in train_ds.column_names if c not in keep_cols])
    val_ds   = val_ds  .map(tok_fn, batched=True,
        remove_columns=[c for c in val_ds.column_names   if c not in keep_cols])

    # torch 格式
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds  .set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds, tok


def get_num_layers(model) -> int:
    return int(getattr(getattr(model, "config", None), "num_hidden_layers", 12))


def build_llrd_optimizer(model, base_lr: float, layer_decay: float, weight_decay: float):
    """判别式微调（LLRD）：高层大学习率、底层小学习率。"""
    no_decay = ["bias", "LayerNorm.weight"]
    n_layers = get_num_layers(model)
    layer_map = {}
    for name, _ in model.named_parameters():
        if "embeddings" in name:
            layer_map[name] = 0; continue
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        if m: layer_map[name] = int(m.group(1)) + 1
        else: layer_map[name] = n_layers + 2  # pooler/cls 等视作高层

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        layer_id = layer_map[name]
        scale = layer_decay ** (n_layers + 2 - layer_id)
        lr = base_lr * scale
        decay = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        key = (lr, decay)
        if key not in groups: groups[key] = {"params": [], "lr": lr, "weight_decay": decay}
        groups[key]["params"].append(p)
    return AdamW(list(groups.values()))


class RDropLoss(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = float(alpha)
        self.kl = nn.KLDivLoss(reduction="batchmean")
    def forward(self, logits1, logits2, labels):
        ce = 0.5 * (
            nn.functional.cross_entropy(logits1, labels) +
            nn.functional.cross_entropy(logits2, labels)
        )
        p1 = nn.functional.log_softmax(logits1, dim=-1); q1 = p1.exp()
        p2 = nn.functional.log_softmax(logits2, dim=-1); q2 = p2.exp()
        kl = 0.5 * (self.kl(p1, q2) + self.kl(p2, q1))
        return ce + self.alpha * kl


class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, logits): return logits / torch.exp(self.logT)
    @torch.no_grad()
    def temperature(self): return float(torch.exp(self.logT).item())


class SWACallback(EarlyStoppingCallback):
    """SWA：按 epoch 平均权重；沿用 EarlyStopping 的耐心值。"""
    def __init__(self, enable: bool):
        super().__init__(early_stopping_patience=2)
        self.enable = enable
        self.swa_model = None
    def on_train_begin(self, args, state, control, **kw):
        if self.enable:
            self.swa_model = AveragedModel(kw["model"])
    def on_epoch_end(self, args, state, control, **kw):
        super().on_epoch_end(args, state, control, **kw)
        if self.enable and self.swa_model is not None:
            self.swa_model.update_parameters(kw["model"])
    def on_train_end(self, args, state, control, **kw):
        if self.enable and self.swa_model is not None:
            tgt, src = kw["model"].state_dict(), self.swa_model.module.state_dict()
            for k in tgt.keys():
                if k in src: tgt[k].data.copy_(src[k].data)


class AdvTrainer(Trainer):
    """集成 FreeLB（嵌入 PGD） + R-Drop（可关）。"""
    def __init__(self, *args,
                 adv_steps: int = 2, adv_epsilon: float = 0.3, adv_alpha: float = 0.1,
                 rdrop_alpha: float = 0.2,
                 use_llrd: bool = True, layer_decay: float = 0.85,
                 **kw):
        super().__init__(*args, **kw)
        self.adv_steps = int(adv_steps)
        self.adv_epsilon = float(adv_epsilon)
        self.adv_alpha = float(adv_alpha)
        self.use_llrd = bool(use_llrd)
        self.layer_decay = float(layer_decay)
        self.rdrop = RDropLoss(alpha=rdrop_alpha) if rdrop_alpha > 0 else None

    def create_optimizer(self):
        if self.optimizer is not None: return
        if self.use_llrd:
            wd = self.args.weight_decay
            self.optimizer = build_llrd_optimizer(self.model, self.args.learning_rate, self.layer_decay, wd)
        else:
            super().create_optimizer()

    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: Optional[int] = None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # -------- natural loss（可含R-Drop）--------
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(loss)

        # -------- FreeLB（保守默认）--------
        if self.adv_steps > 0 and self.adv_epsilon > 0:
            emb: nn.Embedding = model.get_input_embeddings()
            backup = emb.weight.data.clone()

            for _ in range(self.adv_steps):
                grad = emb.weight.grad
                if grad is None: break
                # 步进 + 投影到 |delta|<=eps
                norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-12
                r_at = self.adv_alpha * grad / norm
                emb.weight.data.add_(r_at)
                delta = emb.weight.data - backup
                emb.weight.data = backup + delta.clamp(min=-self.adv_epsilon, max=self.adv_epsilon)

                # 累积对抗梯度
                model.zero_grad(set_to_none=True)
                adv_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(adv_loss / self.adv_steps)

            emb.weight.data = backup  # restore

        return loss.detach()

    # 兼容 4.5x：接受 num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        labels = inputs["labels"]
        if self.rdrop is not None:
            out1 = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            out2 = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits1, logits2 = out1.logits, out2.logits
            loss = self.rdrop(logits1, logits2, labels)
            outputs = {"logits": (logits1 + logits2) * 0.5}
            return (loss, outputs) if return_outputs else loss
        else:
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            return (out.loss, out) if return_outputs else out.loss


def temperature_scaling(trainer: Trainer, val_ds, out_dir: str):
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
        loss = nll(scaler(logits), batch["labels"])
        loss.backward()
        return loss

    for _ in range(25):
        optim.step(closure)

    T = scaler.temperature()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"T": T}, f)
    print(f"[TemperatureScaling] learned T={T:.4f} (Guo et al. 2017)")


def main(
    ds_path: str = "./data/processed/mix_zh",
    out: str = "./artifacts/prefilter",
    backbone: str = "hfl/chinese-roberta-wwm-ext",
    epochs: int = 3, batch_size: int = 64, lr: float = 2e-5,
    warmup_ratio: float = 0.06, weight_decay: float = 0.01,
    # FreeLB（保守默认，来自论文常见区间）
    adv_steps: int = 2, adv_epsilon: float = 0.3, adv_alpha: float = 0.1,
    # R-Drop（弱化默认，可设 0 关闭）
    rdrop_alpha: float = 0.2,
    # LLRD / SWA 开关
    use_llrd: bool = True, layer_decay: float = 0.85,
    use_swa: bool = True,
    # 训练稳定性
    fp16: bool = False, seed: int = 42, max_grad_norm: float = 1.0
):
    set_seed(seed)
    train_ds, val_ds, tok = build_dataset(ds_path, backbone)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2)

    collate = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    # 关键超参一览
    print(f"[cfg] backbone={backbone}  epochs={epochs}  bs={batch_size}  lr={lr}  "
          f"warmup_ratio={warmup_ratio}  wd={weight_decay}  "
          f"FreeLB(steps={adv_steps}, eps={adv_epsilon}, alpha={adv_alpha})  "
          f"RDrop(alpha={rdrop_alpha})  LLRD={use_llrd}(decay={layer_decay})  "
          f"SWA={use_swa}  fp16={fp16}  seed={seed}")

    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        fp16=fp16,
        dataloader_num_workers=2,
        logging_steps=50,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": float((preds == labels).mean())}

    callbacks = [SWACallback(enable=use_swa)]

    trainer = AdvTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        adv_steps=adv_steps, adv_epsilon=adv_epsilon, adv_alpha=adv_alpha,
        rdrop_alpha=rdrop_alpha,
        use_llrd=use_llrd, layer_decay=layer_decay
    )

    # scheduler（warmup 按比例）
    trainer.create_optimizer()
    total_steps = (len(train_ds) // batch_size) * epochs
    warmup = max(1, int(warmup_ratio * total_steps))
    trainer.lr_scheduler = get_cosine_schedule_with_warmup(trainer.optimizer, warmup, total_steps)

    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)

    # ---- 温度缩放 ----
    temperature_scaling(trainer, val_ds, out)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ds-path", default="./data/processed/mix_zh")
    p.add_argument("--out", default="./artifacts/prefilter")
    p.add_argument("--backbone", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adv-steps", type=int, default=2)
    p.add_argument("--adv-eps", type=float, default=0.3)
    p.add_argument("--adv-alpha", type=float, default=0.1)
    p.add_argument("--rdrop-alpha", type=float, default=0.2)
    p.add_argument("--use-llrd", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--layer-decay", type=float, default=0.85)
    p.add_argument("--use-swa", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--fp16", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    a = p.parse_args()
    main(
        ds_path=a.ds_path, out=a.out, backbone=a.backbone,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        warmup_ratio=a.warmup_ratio, weight_decay=a.weight_decay,
        adv_steps=a.adv_steps, adv_epsilon=a.adv_eps, adv_alpha=a.adv_alpha,
        rdrop_alpha=a.rdrop_alpha,
        use_llrd=a.use_llrd, layer_decay=a.layer_decay, use_swa=a.use_swa,
        fp16=a.fp16, seed=a.seed, max_grad_norm=a.max_grad_norm
    )
