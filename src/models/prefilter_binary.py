from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, torch.nn as nn

class Prefilter:
    def __init__(self, name="hfl/chinese-roberta-wwm-ext", path=None, device=None):
        self.tok = AutoTokenizer.from_pretrained(name if path is None else path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name if path is None else path, num_labels=2, problem_type="single_label_classification"
        )
        self.model = self.model.eval().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def predict_proba(self, texts, max_len=256):
        batch = self.tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
        batch = {k:v.to(self.model.device) for k,v in batch.items()}
        with torch.no_grad():
            logits = self.model(**batch).logits
            probs = logits.softmax(-1)  # [:,1] = unsafe
        return probs[:,1].detach().cpu().numpy()
