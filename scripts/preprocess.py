import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/imdb_sample.csv"
PROCESSED_DIR = "data/processed"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError("Raw data file not found")

    df = pd.read_csv(RAW_PATH)

    df["text"] = df["text"].apply(clean_text)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print("✅ Data preprocessing completed")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()
