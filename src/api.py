from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

MODEL_PATH = os.getenv("MODEL_PATH", "model_output")

app = FastAPI()

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class TextInput(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        conf, label = torch.max(probs, dim=1)

    return {
        "label": "positive" if label.item() == 1 else "negative",
        "confidence": round(conf.item(), 3)
    }
