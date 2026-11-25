"""Dataset loading, label normalization, and train/validation/test preparation."""
from __future__ import annotations

from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TASK_LABELS


def normalize_text(value: object) -> str:
    """Normalize whitespace without removing punctuation, casing, or context."""
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split()).strip()


def _normalize_spam_label(value: object) -> int:
    """Map UCI/Hugging Face SMS labels to 0=ham and 1=spam."""
    if isinstance(value, str):
        label = value.strip().lower()
        if label in {"ham", "legitimate", "not spam", "0"}:
            return 0
        if label in {"spam", "1"}:
            return 1
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown SMS label: {value!r}") from exc
    if numeric not in (0, 1):
        raise ValueError(f"Expected binary SMS label 0/1, received {value!r}")
    return numeric


def _clean_frame(frame: pd.DataFrame, text_column: str, label_column: str, task: str) -> pd.DataFrame:
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(
            f"Dataset must contain {text_column!r} and {label_column!r}; "
            f"found columns: {list(frame.columns)}"
        )
    clean = frame[[text_column, label_column]].copy()
    clean.columns = ["text", "label"]
    clean["text"] = clean["text"].map(normalize_text)
    clean = clean[clean["text"].str.len() > 0]
    if task == "spam":
        clean["label"] = clean["label"].map(_normalize_spam_label)
    elif task == "hate":
        clean["label"] = clean["label"].astype(int)
        if not clean["label"].isin([0, 1]).all():
            raise ValueError("Hate dataset labels must be 0 or 1")
    else:
        raise ValueError(f"Unsupported task: {task}")
    return clean.drop_duplicates(subset=["text"]).reset_index(drop=True)


def load_spam_splits() -> Dict[str, pd.DataFrame]:
    """Load UCI SMS Spam Collection and create stratified 80/10/10 splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    dataset = load_dataset("ucirvine/sms_spam", split="train")
    frame = dataset.to_pandas()
    text_col = "sms" if "sms" in frame.columns else "text"
    clean = _clean_frame(frame, text_col, "label", "spam")

    train_df, remainder = train_test_split(
        clean, test_size=0.20, random_state=42, stratify=clean["label"]
    )
    valid_df, test_df = train_test_split(
        remainder, test_size=0.50, random_state=42, stratify=remainder["label"]
    )
    return {
        "train": train_df.reset_index(drop=True),
        "validation": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_hate_splits() -> Dict[str, pd.DataFrame]:
    """Load CardiffNLP TweetEval's official hate train/validation/test splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    dataset = load_dataset("cardiffnlp/tweet_eval", "hate")
    result: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        frame = dataset[split_name].to_pandas()
        result[split_name] = _clean_frame(frame, "text", "label", "hate")
    return result


def load_splits(task: str) -> Dict[str, pd.DataFrame]:
    if task == "spam":
        return load_spam_splits()
    if task == "hate":
        return load_hate_splits()
    raise ValueError("task must be 'spam' or 'hate'")


def summarize_splits(splits: Dict[str, pd.DataFrame], task: str) -> dict:
    labels = TASK_LABELS[task]
    return {
        split: {
            "rows": int(len(frame)),
            "class_counts": {
                labels[int(label)]: int(count)
                for label, count in frame["label"].value_counts().sort_index().items()
            },
        }
        for split, frame in splits.items()
    }
