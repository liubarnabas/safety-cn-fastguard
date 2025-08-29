from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os

def export_prefilter_onnx(out="./artifacts/prefilter_onnx"):
    model = AutoModelForSequenceClassification.from_pretrained("./artifacts/prefilter")
    tok = AutoTokenizer.from_pretrained("./artifacts/prefilter")
    os.makedirs(out, exist_ok=True)
    dummy = tok("示例", return_tensors="pt")
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]),
                      f"{out}/model.onnx", input_names=["input_ids","attention_mask"],
                      output_names=["logits"], dynamic_axes={"input_ids":{0:"b",1:"s"},"attention_mask":{0:"b",1:"s"}}, opset_version=17)

if __name__=="__main__": export_prefilter_onnx()
