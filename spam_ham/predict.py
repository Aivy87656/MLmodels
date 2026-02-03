import argparse
from pathlib import Path

import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam or ham for a message.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model.joblib"),
        help="Path to a trained model (joblib).",
    )
    parser.add_argument("--text", type=str, required=True, help="Message to classify.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    pipeline = joblib.load(args.model)
    prediction = pipeline.predict([args.text])[0]
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([args.text])[0]
        labels = pipeline.classes_
        confidence = dict(zip(labels, proba))
    else:
        confidence = {}

    print("Prediction:", prediction)
    if confidence:
        ordered = sorted(confidence.items(), key=lambda item: item[1], reverse=True)
        print("Confidence:")
        for label, score in ordered:
            print(f"- {label}: {score:.2f}")


if __name__ == "__main__":
    main()
