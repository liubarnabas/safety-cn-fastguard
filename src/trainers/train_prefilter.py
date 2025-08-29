from transformers import (
    TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import evaluate, numpy as np
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    f1 = evaluate.load("f1")
    acc = evaluate.load("accuracy")
    return {"f1": f1.compute(predictions=preds, references=labels)["f1"],
            "acc": acc.compute(predictions=preds, references=labels)["accuracy"]}

def main(ds_path="./data/processed/mix_zh", out="./artifacts/prefilter"):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 静默 tokenizer fork 警告

    dsd = load_from_disk(ds_path)
    ds = dsd["train"]

    # SAFE -> 0, 其他 -> 1
    def binlab(e):
        return {"label": 0 if e["labels"] == "SAFE" else 1}
    ds = ds.map(binlab)

    name = "hfl/chinese-roberta-wwm-ext"
    tok = AutoTokenizer.from_pretrained(name)

    # 动态 padding：仅保留需要的列（先保留 text_in + label，map 后再自动替换为 input_ids/attention_mask + label）
    keep_cols = ["text_in", "label"]
    remove_cols = [c for c in ds.column_names if c not in keep_cols]

    def tok_fn(batch):
        return tok(batch["text_in"], truncation=True, max_length=256)
    ds = ds.map(tok_fn, batched=True, remove_columns=remove_cols)

    # 现在 ds 只包含：input_ids, attention_mask, label
    data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

    args = TrainingArguments(
        out,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=True,
        dataloader_num_workers=2,   # 如仍报 tokenizer fork 警告，可暂时降为 0
        logging_steps=50,
        save_total_limit=2,
    )

    tr = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,   # ★ 关键：动态 padding
        compute_metrics=None           # 这里只有 train，没有 eval，就先不评测
    )
    tr.train()
    tr.save_model(out)

if __name__ == "__main__":
    main()
