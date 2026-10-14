"""Train the first zero-path classifiers on a saved dataset.

This is deliberately a small, interpretable baseline. It answers whether a
linear combination of the individual bond values contains predictive signal.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]


def load_dataset(path):
    """Load X and y, rejecting malformed or single-class datasets."""
    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]
        metadata = json.loads(data["metadata"].item())
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("Expected X with shape (samples, features) and y with shape (samples,).")
    if not np.isin(X, [-1, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected ±1 bond features and binary labels.")
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires examples from both classes.")
    return X, y, metadata


def remove_duplicate_configurations(X, y):
    """Keep one copy of each configuration and reject inconsistent labels."""
    unique_X, first, inverse, counts = np.unique(
        X, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    for group in range(len(unique_X)):
        if len(np.unique(y[inverse == group])) != 1:
            raise ValueError("The same bond configuration has conflicting labels.")
    unique_y = y[first]
    return unique_X, unique_y, int(np.sum(counts - 1))


def score_model(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def train_baselines(X, y, test_size=0.2, seed=42):
    """Fit models on one stratified split and return models plus measurements."""
    X, y, duplicates_removed = remove_duplicate_configurations(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    dummy = DummyClassifier(strategy="most_frequent")
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )
    dummy.fit(X_train, y_train)
    logistic.fit(X_train, y_train)

    results = {
        "seed": seed,
        "test_fraction": test_size,
        "duplicates_removed_before_split": duplicates_removed,
        "unique_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_class_counts": np.bincount(y_train, minlength=2).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=2).tolist(),
        "dummy": score_model(dummy, X_test, y_test),
        "logistic_regression": score_model(logistic, X_test, y_test),
    }
    return logistic, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/raw/dataset_L4_demo.npz")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=PROJECT / "models/logistic_L4_demo.joblib")
    parser.add_argument("--results-output", type=Path, default=PROJECT / "models/logistic_L4_demo_metrics.json")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must lie between 0 and 1")
    X, y, metadata = load_dataset(args.dataset)
    model, results = train_baselines(X, y, args.test_size, args.seed)
    results["dataset"] = str(args.dataset.resolve())
    results["dataset_metadata"] = metadata
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.results_output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Saved model to {args.model_output}")


if __name__ == "__main__":
    main()
