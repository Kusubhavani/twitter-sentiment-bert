import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import argparse

def main(input_file, output_file):
    tokenizer = DistilBertTokenizerFast.from_pretrained("model_output")
    model = DistilBertForSequenceClassification.from_pretrained("model_output")
    model.eval()

    df = pd.read_csv(input_file)

    sentiments = []
    confidences = []

    for text in df["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            conf, label = torch.max(probs, dim=1)

        sentiments.append("positive" if label.item() == 1 else "negative")
        confidences.append(round(conf.item(), 3))

    df["sentiment"] = sentiments
    df["confidence"] = confidences
    df.to_csv(output_file, index=False)

    print("✅ Batch prediction completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    main(args.input_file, args.output_file)
