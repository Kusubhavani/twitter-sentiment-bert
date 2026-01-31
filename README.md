# Twitter / IMDB Sentiment Analysis using BERT

This project implements an **end-to-end Sentiment Analysis system** using a **pretrained BERT (DistilBERT)** model.
The system supports **training, batch prediction, and real-time inference** through a **FastAPI service**, fully containerized using **Docker and Docker Compose**.

---

## 🚀 Features

* Text sentiment classification (Positive / Negative)
* Fine-tuning **DistilBERT** using Hugging Face Transformers
* Batch prediction on CSV files
* REST API built with **FastAPI**
* Fully Dockerized setup
* One-command startup using `docker-compose`

---

## 🧱 Project Structure

```
twitter-sentiment-bert/
│
├── data/
│   ├── raw/                # Raw dataset (imdb_sample.csv)
│   └── processed/          # Preprocessed train/test data
│
├── scripts/
│   ├── preprocess.py       # Data preprocessing
│   ├── train.py            # Model training
│   └── batch_predict.py    # Batch inference on CSV
│
├── src/
│   └── api.py              # FastAPI application
│
├── model_output/           # Trained model artifacts
├── results/                # Prediction outputs
│
├── Dockerfile.api          # Dockerfile for FastAPI service
├── docker-compose.yml      # Multi-service orchestration
├── requirements.api.txt   # API dependencies
├── .env.example            # Environment variables template
└── README.md
```

---

## 📊 Dataset

* **IMDB Movie Reviews Dataset**
* Format: CSV
* Columns:

  * `text` → review text
  * `label` → sentiment label (0 = negative, 1 = positive)

Sample file:

```
data/raw/imdb_sample.csv
```

---

## ⚙️ Local Setup (Optional – Without Docker)

```bash
Open the powershell
clone the git by using :
git clone https://github.com/Kusubhavani/twitter-sentiment-bert

cd twitter-sentiment-bert
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.api.txt
```

### Preprocess Data

```bash
python scripts/preprocess.py
```

### Train Model

```bash
python scripts/train.py
```

### Batch Prediction

```bash
python scripts/batch_predict.py \
  --input-file data/raw/imdb_sample.csv \
  --output-file results/imdb_predictions.csv
```

---

## 🐳 Run with Docker (Recommended)

### Prerequisites

* Docker
* Docker Compose

### Start the Application

```bash
docker-compose up --build
```

This command:

* Builds the Docker image
* Installs dependencies
* Starts the FastAPI service automatically

---

## 🌐 API Usage

Once running, access:

### Swagger UI

```
http://localhost:8000/docs
```

### Health Check

```
GET /health
```

Response:

```json
"ok"
```

### Sentiment Prediction

```
POST /predict
```

**Request Body**

```json
{
  "text": "This movie was amazing!"
}
```

**Response**

```json
{
  "sentiment": "positive",
  "confidence": 0.87
}
```

---

## 📦 Environment Variables

All environment variables are documented in:

```
.env.example
```

Example:

```env
MODEL_PATH=model_output
```

---

## 🧪 Model Details

* Model: `distilbert-base-uncased`
* Framework: Hugging Face Transformers
* Training: Fine-tuned for binary sentiment classification
* Output: Label + confidence score

---

## ✅ Submission Checklist

* [x] Dockerized application
* [x] docker-compose.yml at root
* [x] FastAPI service running automatically
* [x] Swagger UI available
* [x] README with full instructions
* [x] `.env.example` included

---

## 👤 Author

**Bhavani**

