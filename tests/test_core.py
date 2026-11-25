import pandas as pd
import pytest

from src.data import normalize_text, _normalize_spam_label, _clean_frame
from src.demo import demo_predict
from src.config import TASK_LABELS


def test_normalize_text_collapses_whitespace_and_preserves_punctuation():
    assert normalize_text("  Hello,\n  world!  ") == "Hello, world!"


def test_normalize_text_handles_none():
    assert normalize_text(None) == ""


@pytest.mark.parametrize("value,expected", [("ham", 0), ("spam", 1), (0, 0), (1, 1), ("1", 1)])
def test_spam_label_normalization(value, expected):
    assert _normalize_spam_label(value) == expected


def test_spam_label_rejects_unknown_value():
    with pytest.raises(ValueError):
        _normalize_spam_label("unknown")


def test_clean_frame_drops_empty_text_and_duplicates():
    raw = pd.DataFrame({"sms": ["Hello", " Hello ", "  "], "label": ["ham", "ham", "spam"]})
    clean = _clean_frame(raw, "sms", "label", "spam")
    assert len(clean) == 1
    assert clean.iloc[0]["text"] == "Hello"
    assert int(clean.iloc[0]["label"]) == 0


def test_demo_spam_rule_flags_obvious_prize_link():
    result = demo_predict("spam", "Congratulations! Claim your free prize at https://example.com")
    assert result["label"] == "spam"
    assert result["source"].startswith("DEMO RULES")


def test_demo_hate_rule_flags_group_targeted_hostility():
    result = demo_predict("hate", "I hate all immigrants and they should be forced out.")
    assert result["label"] == "hate"


def test_demo_hate_rule_does_not_flag_neutral_mention():
    result = demo_predict("hate", "The conference included speakers from India and Kenya.")
    assert result["label"] == "non-hate"


def test_demo_threshold_changes_decision():
    result = demo_predict("spam", "URGENT! Claim your prize now", threshold=0.95)
    assert result["label"] == "ham"
    assert result["positive_probability"] < 0.95


def test_expected_label_mappings_are_binary():
    assert TASK_LABELS["spam"] == {0: "ham", 1: "spam"}
    assert TASK_LABELS["hate"] == {0: "non-hate", 1: "hate"}
