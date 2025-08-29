#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel


class MultiLabelMDeberta(nn.Module):
    """
    mDeBERTa-v3-base 作为编码器，CLS -> 13 维多标签（S1..S13）
    """
    def __init__(self, name: str = "microsoft/mdeberta-v3-base", num_labels: int = 13):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls))
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        return {"loss": loss, "logits": logits}
