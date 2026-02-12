"""Train a Random Forest model for static sign language classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sign language classifier.")
    parser.add_argument("--data", type=Path, default=Path("dataset/landmarks.csv"), help="Path to CSV dataset.")
    parser.add_argument("--model-out", type=Path, default=Path("models/sign_rf.joblib"), help="Output model file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if "label" not in data.dtype.names:
        raise ValueError("Dataset must contain a 'label' column.")

    feature_cols = [c for c in data.dtype.names if c != "label"]
    X = np.column_stack([data[c].astype(np.float32) for c in feature_cols])
    y_text = data["label"].astype(str)
    return X, y_text


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    X, y_text = load_dataset(args.data)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Random Forest provides good performance with low inference complexity.
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "encoder": encoder}, args.model_out)
    print(f"Model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
