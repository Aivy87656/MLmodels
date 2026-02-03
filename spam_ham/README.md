# Spam vs. Ham Classifier

This project provides a lightweight spam/ham text classifier using TF-IDF features and a Naive Bayes model.
It includes a tiny sample dataset, a training script, and a prediction script.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
cd spam_ham
python train.py --data data/sample.csv --model-out model.joblib
```

## Predict

```bash
cd spam_ham
python predict.py --model model.joblib --text "Congratulations! You have won a prize."
```

## Dataset Format

CSV with columns:

- `label` (spam or ham)
- `text` (message content)
