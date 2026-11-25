"""Batch and single-text inference for locally fine-tuned models."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable
import numpy as np
import torch

from .config import MAX_LENGTH, MODEL_DIRS, TASK_LABELS


class ModelNotFoundError(FileNotFoundError):
    """Raised when a task's fine-tuned model has not been trained/downloaded."""


@lru_cache(maxsize=2)
def _load_model(task: str):
    if task not in MODEL_DIRS:
        raise ValueError("task must be 'spam' or 'hate'")
    model_path = MODEL_DIRS[task]
    if not (model_path / "config.json").exists():
        raise ModelNotFoundError(
            f"No fine-tuned {task} model found at {model_path}. "
            f"Train it with: python -m src.train --task {task}"
        )
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install model dependencies: pip install -r requirements.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def model_available(task: str) -> bool:
    return (MODEL_DIRS[task] / "config.json").exists()


def predict_many(task: str, texts: Iterable[str], threshold: float = 0.5, batch_size: int = 16) -> list[dict]:
    """Return positive-class probabilities, labels, and confidence per text."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    normalized_texts = [" ".join(str(text or "").split()) for text in texts]
    if any(not text for text in normalized_texts):
        raise ValueError("Text inputs must not be empty")
    if not normalized_texts:
        return []

    tokenizer, model, device = _load_model(task)
    results: list[dict] = []
    for start in range(0, len(normalized_texts), batch_size):
        batch = normalized_texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        for text, probs in zip(batch, probabilities):
            positive_probability = float(probs[1])
            predicted_id = int(positive_probability >= threshold)
            class_probability = float(probs[predicted_id])
            results.append({
                "text": text,
                "label_id": predicted_id,
                "label": TASK_LABELS[task][predicted_id],
                "positive_probability": positive_probability,
                "confidence": class_probability,
                "class_probabilities": {
                    TASK_LABELS[task][0]: float(probs[0]),
                    TASK_LABELS[task][1]: float(probs[1]),
                },
                "threshold": float(threshold),
                "source": "fine-tuned DistilBERT",
            })
    return results


def predict_one(task: str, text: str, threshold: float = 0.5) -> dict:
    return predict_many(task, [text], threshold=threshold)[0]
