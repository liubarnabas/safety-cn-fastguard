#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线二分类骨干 A/B 训练脚本（Accuracy 优先）：
  --backbone ∈ {roberta, macbert, deberta_zh}
  统一：早停 + LLRD(可关) + 温度缩放(全量验证集) + 最佳阈值搜索（在验证集上最大化 Accuracy）。
  可选导出 ONNX（后续可做 INT8 动态量化以降延迟）。

兼容说明：
- 针对 Transformers 老版本（不支持 evaluation_strategy 等参数），本脚本自动降级为 eval_strategy 或 steps 逻辑。
- 针对 create_scheduler / create_optimizer_and_scheduler 的版本差异，自动选择可用接口。
"""

import os, sys, json, re, inspect
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)

# --- 早停兼容处理：部分老版本无 EarlyStoppingCallback 时降级为无回调 ---
try:
    from transformers import EarlyStoppingCallback
except Exception:
    class EarlyStoppingCallback:  # type: ignore
        def __init__(self, *args, **kwargs): ...
        def on_evaluate(self, *args, **kwargs): ...

# --- src 布局 ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BACKBONES = {
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "deberta_zh": "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese",
}

def map_binary_label(ex: Dict[str, Any]) -> Dict[str, Any]:
    return {"label": 0 if ex["labels"] == "SAFE" else 1}

def build_dataset(ds_path: str, backbone_name: str, max_len: int = 256):
    dsd = load_from_disk(ds_path)
    ds = dsd["train"].map(map_binary_label)
    dss = ds.train_test_split(test_size=0.12, seed=42)
    train_ds, val_ds = dss["train"], dss["test"]

    tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
    keep_cols = ["text_in", "label"]
    def tok_fn(b): return tok(b["text_in"], truncation=True, max_length=max_len)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=[c for c in train_ds.column_names if c not in keep_cols])
    val_ds   = val_ds  .map(tok_fn, batched=True, remove_columns=[c for c in val_ds.column_names   if c not in keep_cols])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return train_ds, val_ds, tok

def get_num_layers(model) -> int:
    return int(getattr(getattr(model, "config", None), "num_hidden_layers", 12))

def build_llrd_optimizer(model, base_lr: float, layer_decay: float, weight_decay: float):
    from torch.optim import AdamW
    no_decay = ["bias", "LayerNorm.weight"]
    n_layers = get_num_layers(model)
    layer_map = {}
    for name, _ in model.named_parameters():
        if "embeddings" in name:
            layer_map[name] = 0
            continue
        m = re.search(r"encoder\.layer\.(\d+)\.", name)
        if m:
            layer_map[name] = int(m.group(1)) + 1
        else:
            layer_map[name] = n_layers + 2
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

class MyTrainer(Trainer):
    # 兼容部分版本的 num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

@torch.no_grad()
def collect_val_logits_labels(trainer: Trainer, val_ds):
    model = trainer.model.eval()
    device = next(model.parameters()).device
    dl = trainer.get_eval_dataloader(val_ds)
    all_logits, all_labels = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        all_logits.append(logits.float().cpu())
        all_labels.append(batch["labels"].long().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)

def temperature_scaling(trainer: Trainer, val_ds, out_dir: str):
    class TemperatureScaler(nn.Module):
        def __init__(self): super().__init__(); self.logT = nn.Parameter(torch.zeros(1))
        def forward(self, logits): return logits / torch.exp(self.logT)
        @torch.no_grad()
        def temperature(self): return float(torch.exp(self.logT).item())

    logits, labels = collect_val_logits_labels(trainer, val_ds)
    scaler = TemperatureScaler()
    nll = nn.CrossEntropyLoss()
    optim = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=100, tolerance_grad=1e-8, tolerance_change=1e-9)

    def closure():
        optim.zero_grad(set_to_none=True)
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    scaler.train()
    for _ in range(10):
        optim.step(closure)

    T = scaler.temperature()
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"T": T}, f)
    print(f"[TemperatureScaling] learned T={T:.4f}")
    return T

@torch.no_grad()
def search_best_threshold_acc(trainer: Trainer, val_ds, out_dir: str, T: float = 1.0):
    logits, labels = collect_val_logits_labels(trainer, val_ds)
    if T != 1.0:
        logits = logits / T
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    y = labels.cpu().numpy()
    grid = np.linspace(0.05, 0.95, 181)  # 步长 0.005
    best_theta, best_acc = 0.5, -1.0
    for th in grid:
        pred = (probs >= th).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_theta, best_acc = float(th), float(acc)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "theta.json"), "w", encoding="utf-8") as f:
        json.dump({"theta_unsafe": best_theta, "val_acc": best_acc}, f)
    print(f"[Val Threshold] best theta_unsafe={best_theta:.3f}  val_acc={best_acc:.4f}")
    return best_theta, best_acc

def export_onnx(out_dir: str, max_len: int = 256, onnx_name: str = "model.onnx"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).eval()
    tok = AutoTokenizer.from_pretrained(out_dir, use_fast=True)

    dummy = tok(["示例"], return_tensors="pt", truncation=True, max_length=max_len)
    dynamic_axes = {"input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch"}}
    onnx_path = os.path.join(out_dir, onnx_name)
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]),
        onnx_path, input_names=["input_ids","attention_mask"], output_names=["logits"],
        dynamic_axes=dynamic_axes, opset_version=17, do_constant_folding=True
    )
    print(f"[ONNX] exported to {onnx_path}")

def _build_training_arguments(out: str, batch_size: int, lr: float, epochs: int):
    """
    为不同 Transformers 版本构建兼容的 TrainingArguments。
    - 新版本：使用 evaluation_strategy/save_strategy/warmup_ratio 等
    - 旧版本：降级为 eval_strategy 或使用 steps 控制
    """
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters.keys()

    ta_kwargs = dict(
        output_dir=out,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        fp16=True,
        logging_steps=50,
        save_total_limit=3,
        report_to="none",
        weight_decay=0.01,
        seed=42,
        dataloader_num_workers=2
    )

    # eval/evaluation_strategy 兼容
    if "evaluation_strategy" in params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        ta_kwargs["eval_strategy"] = "epoch"
    # save_strategy 兼容
    if "save_strategy" in params:
        ta_kwargs["save_strategy"] = "epoch"
    else:
        # 老版本只有 steps：给一个较大的 save_steps，避免过于频繁保存
        ta_kwargs["save_steps"] = 1000000

    # load_best_model_at_end / metric_for_best_model / greater_is_better 兼容
    if "load_best_model_at_end" in params:
        ta_kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in params:
        ta_kwargs["metric_for_best_model"] = "accuracy"
    if "greater_is_better" in params:
        ta_kwargs["greater_is_better"] = True

    # eval batch size（部分老版本没有该参数）
    if "per_device_eval_batch_size" in params:
        ta_kwargs["per_device_eval_batch_size"] = max(32, batch_size)

    # warmup（优先 ratio，不支持则用 steps）
    if "warmup_ratio" in params:
        ta_kwargs["warmup_ratio"] = 0.06
    elif "warmup_steps" in params:
        ta_kwargs["warmup_steps"] = 0  # 由我们后面手动设定 scheduler 时再处理

    return TrainingArguments(**ta_kwargs)

def _create_or_patch_sched(trainer: Trainer, train_len: int, batch_size: int, epochs: int, use_llrd: bool):
    """
    针对不同版本的 Trainer，创建或补齐学习率调度器。
    - 若使用 LLRD（自定义 optimizer），需要我们来创建 scheduler。
    - 尽量使用 create_scheduler；若无则回退到 get_linear_schedule_with_warmup。
    """
    total_steps = max(1, ( (train_len + batch_size - 1) // batch_size ) * epochs)
    warmup = int(0.06 * total_steps)

    if use_llrd:
        # 优先新版接口
        if hasattr(trainer, "create_scheduler"):
            try:
                trainer.create_scheduler(num_training_steps=total_steps)
                # 如果是线性调度且无 warmup，可尝试设置到相近效果（不同版本行为不同，此处以存在即用）
            except Exception:
                pass
        else:
            # 老接口：手动创建线性 warmup 调度
            try:
                from transformers import get_linear_schedule_with_warmup
                trainer.lr_scheduler = get_linear_schedule_with_warmup(trainer.optimizer, warmup, total_steps)
            except Exception:
                # 最后兜底：不设 scheduler
                trainer.lr_scheduler = None
    else:
        # 不使用 LLRD，交给 HF 内部逻辑创建；如必要也可补齐
        if not hasattr(trainer, "lr_scheduler") or trainer.lr_scheduler is None:
            try:
                if hasattr(trainer, "create_scheduler"):
                    trainer.create_scheduler(num_training_steps=total_steps)
            except Exception:
                try:
                    from transformers import get_linear_schedule_with_warmup
                    trainer.lr_scheduler = get_linear_schedule_with_warmup(trainer.optimizer, warmup, total_steps)
                except Exception:
                    trainer.lr_scheduler = None

def main(
    ds_path: str = "./data/processed/binary_zh",
    out: str = "./artifacts/prefilter",
    which: str = "roberta",
    epochs: int = 3, batch_size: int = 64, lr: float = 2e-5,
    layer_decay: float = 0.85, use_llrd: bool = True,
    export_onnx_flag: bool = False
):
    assert which in BACKBONES, f"--backbone must be one of {list(BACKBONES.keys())}"
    backbone = BACKBONES[which]
    train_ds, val_ds, tok = build_dataset(ds_path, backbone)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=2)
    collate = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    args = _build_training_arguments(out, batch_size, lr, epochs)

    def compute_metrics(eval_pred):
        # 兼容不同 HF 版本的 EvalPrediction 结构
        if isinstance(eval_pred, tuple):
            logits, labels = eval_pred
        else:
            logits, labels = eval_pred.predictions, eval_pred.label_ids
        preds = logits.argmax(-1)
        import numpy as _np
        return {"accuracy": float((_np.asarray(preds) == _np.asarray(labels)).mean())}

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if EarlyStoppingCallback is not object else []
    trainer = MyTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate, compute_metrics=compute_metrics, callbacks=callbacks
    )

    # 优化器/调度器
    if use_llrd:
        trainer.optimizer = build_llrd_optimizer(model, lr, layer_decay, weight_decay=0.01)
        _create_or_patch_sched(trainer, train_len=len(train_ds), batch_size=batch_size, epochs=epochs, use_llrd=True)
    else:
        # 让 HF 自己创建；必要时再补齐
        try:
            if hasattr(trainer, "create_optimizer") and hasattr(trainer, "create_scheduler"):
                trainer.create_optimizer()
                trainer.create_scheduler(num_training_steps=None)
            else:
                trainer.create_optimizer_and_scheduler(num_training_steps=None)  # 旧版本
        except Exception:
            _create_or_patch_sched(trainer, train_len=len(train_ds), batch_size=batch_size, epochs=epochs, use_llrd=False)

    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)

    # 温度缩放 + 阈值搜索（保存 T 和 θ，线上直接用）
    T = temperature_scaling(trainer, val_ds, out)
    search_best_threshold_acc(trainer, val_ds, out, T=T)

    # 可选：导出 ONNX
    if export_onnx_flag:
        export_onnx(out_dir=out, max_len=256)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ds-path", default="./data/processed/binary_zh")
    p.add_argument("--out", default="./artifacts/prefilter")
    p.add_argument("--backbone", choices=list(BACKBONES.keys()), default="roberta")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--layer-decay", type=float, default=0.85)
    p.add_argument("--use-llrd", type=lambda x: str(x).lower() in {"1","true","yes"}, default=True)
    p.add_argument("--export-onnx", type=lambda x: str(x).lower() in {"1","true","yes"}, default=False)
    a = p.parse_args()
    main(
        ds_path=a.ds_path, out=a.out, which=a.backbone, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, layer_decay=a.layer_decay,
        use_llrd=a.use_llrd, export_onnx_flag=a.export_onnx
    )
