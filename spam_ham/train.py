import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    data = pd.read_csv(csv_path)
    required_columns = {"label", "text"}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing))}")
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("model", MultinomialNB()),
        ]
    )


def train_model(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )
    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    return pipeline, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a spam/ham classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/sample.csv"),
        help="Path to a CSV file with columns: label,text",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("model.joblib"),
        help="Output path for the trained model.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    pipeline, report = train_model(data, args.test_size, args.random_state)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    print("Model saved to", args.model_out)
    print("\nClassification report:")
    for label, metrics in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1-score", 0)
        support = metrics.get("support", 0)
        print(f"- {label}: precision={precision:.2f} recall={recall:.2f} f1={f1:.2f} support={support}")
    print(f"Accuracy: {report.get('accuracy', 0):.2f}")


if __name__ == "__main__":
    main()
