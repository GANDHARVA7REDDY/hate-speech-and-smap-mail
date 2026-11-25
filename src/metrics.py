"""Metrics shared by training and post-training evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(eval_prediction) -> dict:
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average="binary", pos_label=1, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
    }
