"""Evaluate a saved fine-tuned model on its held-out test split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from .config import MODEL_DIRS, TASK_LABELS
from .data import load_splits
from .predict import predict_many


def evaluate(task: str, threshold: float = 0.5) -> dict:
    splits = load_splits(task)
    test = splits["test"]
    predictions = predict_many(task, test["text"].tolist(), threshold=threshold)
    y_true = test["label"].astype(int).to_numpy()
    y_pred = np.asarray([item["label_id"] for item in predictions], dtype=int)
    labels = TASK_LABELS[task]
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=[labels[0], labels[1]],
        output_dict=True, zero_division=0,
    )
    metrics = {
        "task": task,
        "threshold": threshold,
        "rows": int(len(test)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
    }
    target = MODEL_DIRS[task] / "latest_evaluation.json"
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved report: {target}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["spam", "hate"], required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.task, args.threshold)


if __name__ == "__main__":
    main()
