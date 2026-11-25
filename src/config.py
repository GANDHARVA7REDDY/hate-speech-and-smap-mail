"""Shared project configuration."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIRS = {
    "spam": ARTIFACTS_DIR / "spam_model",
    "hate": ARTIFACTS_DIR / "hate_model",
}
TASK_LABELS = {
    "spam": {0: "ham", 1: "spam"},
    "hate": {0: "non-hate", 1: "hate"},
}
TASK_DISPLAY_NAMES = {"spam": "Spam detection", "hate": "Hate speech detection"}
DEFAULT_BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 192
