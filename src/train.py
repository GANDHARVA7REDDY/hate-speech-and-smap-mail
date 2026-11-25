"""Fine-tune a DistilBERT binary classifier for spam and/or hate speech.

Examples:
    python -m src.train --task spam --epochs 3
    python -m src.train --task hate --epochs 3
    python -m src.train --task all --epochs 3
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .config import ARTIFACTS_DIR, DEFAULT_BASE_MODEL, MAX_LENGTH, MODEL_DIRS, TASK_LABELS
from .data import load_splits, summarize_splits
from .metrics import classification_metrics


def _make_training_arguments(TrainingArguments, output_dir: Path, epochs: int, batch_size: int):
    """Use the evaluation-strategy parameter name supported by installed Transformers."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": epochs,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(batch_size, 8),
        "logging_steps": 25,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": bool(torch.cuda.is_available()),
        "dataloader_pin_memory": bool(torch.cuda.is_available()),
        "seed": 42,
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    supported = set(params)
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in supported})


def train_one_task(task: str, model_name: str, epochs: int, batch_size: int) -> dict:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("Missing ML libraries. Run: pip install -r requirements.txt") from exc

    output_dir = MODEL_DIRS[task]
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(task)
    print(f"\n[{task.upper()}] Dataset splits:")
    print(json.dumps(summarize_splits(splits, task), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label_names = TASK_LABELS[task]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=label_names,
        label2id={name: idx for idx, name in label_names.items()},
    )

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = {}
    for split_name, frame in splits.items():
        data = Dataset.from_pandas(
            frame[["text", "label"]].rename(columns={"label": "labels"}),
            preserve_index=False,
        )
        tokenized[split_name] = data.map(tokenize_batch, batched=True, remove_columns=["text"])

    train_labels = np.asarray(splits["train"]["label"], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Training data for {task} is missing one class: {class_counts.tolist()}")
    weights = len(train_labels) / (2.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            labels = inputs["labels"]
            weights_on_device = self.class_weights.to(outputs.logits.device)
            loss = torch.nn.CrossEntropyLoss(weight=weights_on_device)(
                outputs.logits.view(-1, 2), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    args = _make_training_arguments(TrainingArguments, output_dir, epochs, batch_size)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=classification_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_output = trainer.predict(tokenized["test"])
    predicted = np.argmax(test_output.predictions, axis=-1)
    actual = np.asarray(splits["test"]["label"], dtype=np.int64)
    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=[label_names[0], label_names[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics = classification_metrics(type("EvalPrediction", (), {
        "predictions": test_output.predictions,
        "label_ids": test_output.label_ids,
    })())
    metrics["confusion_matrix"] = confusion_matrix(actual, predicted, labels=[0, 1]).tolist()
    metrics["classification_report"] = report
    metrics["dataset_summary"] = summarize_splits(splits, task)
    metrics["base_model"] = model_name
    metrics["task"] = task
    metrics["epochs_requested"] = epochs
    metrics["max_length"] = MAX_LENGTH
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"\n[{task.upper()}] Test metrics:")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "dataset_summary"}}, indent=2))
    print(f"Saved fine-tuned model to: {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT text classifiers")
    parser.add_argument("--task", choices=["spam", "hate", "all"], default="all")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Hugging Face base model ID")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive integers")
    tasks = ["spam", "hate"] if args.task == "all" else [args.task]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        train_one_task(task, args.model, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
