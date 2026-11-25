"""Transparent rule-based preview used only before fine-tuned models exist.

These heuristics are for checking the Streamlit workflow, not for moderation decisions.
"""
from __future__ import annotations
import re
from .config import TASK_LABELS

SPAM_PATTERNS = [
    r"https?://|www\.", r"\bclaim\b", r"\bwon\b|\bwinner\b|\bprize\b",
    r"\bfree\b", r"\burgent\b", r"\bverify your (account|details|password)",
    r"\bclick here\b", r"\blimited time\b", r"\bloan approved\b",
    r"\bcongratulations\b", r"\bcall now\b", r"\bOTP\b",
]
HATE_PATTERNS = [
    r"\ball\s+[a-z][a-z -]{1,35}\s+(are|should|must)\b",
    r"\b(drive|kick|force)\s+(them|those people|all)\s+out\b",
    r"\b(should|must)\s+be\s+(driven|kicked|forced)\s+out\b",
    r"\b(hate|despise)\s+(all\s+)?(immigrants|foreigners|women|men|muslims|christians|hindus|jews|disabled people|gay people)\b",
    r"\b(inferior|vermin|subhuman)\b",
]


def demo_predict(task: str, text: str, threshold: float = 0.5) -> dict:
    clean = " ".join(str(text or "").split())
    if not clean:
        raise ValueError("Please enter a message")
    lowered = clean.lower()
    patterns = SPAM_PATTERNS if task == "spam" else HATE_PATTERNS if task == "hate" else None
    if patterns is None:
        raise ValueError("task must be 'spam' or 'hate'")
    matches = sum(bool(re.search(pattern, lowered, flags=re.IGNORECASE)) for pattern in patterns)
    positive_probability = min(0.94, 0.22 + 0.16 * matches) if matches else 0.12
    predicted_id = int(positive_probability >= threshold)
    probs = [1.0 - positive_probability, positive_probability]
    return {
        "text": clean,
        "label_id": predicted_id,
        "label": TASK_LABELS[task][predicted_id],
        "positive_probability": positive_probability,
        "confidence": probs[predicted_id],
        "class_probabilities": {
            TASK_LABELS[task][0]: probs[0],
            TASK_LABELS[task][1]: probs[1],
        },
        "threshold": threshold,
        "source": "DEMO RULES — NOT A TRAINED MODEL",
    }
