# Model training and fine-tuning

## Model choice

The project fine-tunes `distilbert-base-uncased`, a compact English transformer encoder, separately for each task. It starts from pretrained language representations and learns a task-specific two-class output head.

**Input:** raw message text  
**Tokenization:** Hugging Face AutoTokenizer, truncation enabled, maximum length 192  
**Output:** class logits converted to probabilities with softmax  
**Positive classes:** `spam` for the spam model and `hate` for the hate model

## Setup

Use Python 3.10–3.12 for the smoothest package compatibility. A GPU is helpful but not required. Fine-tuning on CPU is possible but can be slow.

```bash
# create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails or you need a platform-specific build, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Train both models

```bash
python -m src.train --task all --epochs 3 --batch-size 8
```

Train only one classifier:

```bash
python -m src.train --task spam --epochs 3 --batch-size 8
python -m src.train --task hate --epochs 3 --batch-size 8
```

The first run downloads the base model and datasets. This requires an internet connection and enough disk space. The default base model is `distilbert-base-uncased`; override it with `--model` if needed.

### Useful arguments

| Argument | Default | Description |
|---|---:|---|
| `--task` | `all` | `spam`, `hate`, or `all` |
| `--model` | `distilbert-base-uncased` | Hugging Face base model ID |
| `--epochs` | `3` | Number of fine-tuning epochs |
| `--batch-size` | `8` | Per-device training batch size |

For a quick smoke experiment, start with one task and one epoch. Do not use a tiny smoke run as a final model or report its metrics as production-quality results.

## Fine-tuning details

- Optimizer and learning-rate scheduling are handled by Hugging Face `Trainer` defaults.
- Learning rate: `2e-5`.
- Weight decay: `0.01`.
- Batch size: configurable; defaults to 8 per device.
- Class-weighted cross-entropy is used to reduce the influence of imbalanced class counts.
- Evaluation runs after each epoch; the checkpoint with best positive-class F1 is restored.
- A fixed seed (`42`) supports repeatability, but hardware and library differences can still produce variation.

## Output artifacts

```text
artifacts/
├── spam_model/
│   ├── config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   ├── tokenizer files...
│   ├── evaluation_metrics.json
│   └── checkpoints/
└── hate_model/
    └── ...
```

The final test report includes accuracy, positive-class precision, recall, F1, macro-F1, a confusion matrix, and a class-wise classification report. The test set should be used once for final evaluation, not repeatedly for tuning.

## Evaluate saved model again

```bash
python -m src.evaluate --task spam --threshold 0.50
python -m src.evaluate --task hate --threshold 0.50
```

This evaluates the saved model on the test split and writes `latest_evaluation.json` inside the model folder.

## Threshold tuning

The Streamlit interface has independent thresholds for each model. Raising the positive threshold generally reduces positive predictions and may improve precision while lowering recall; lowering it generally increases recall while potentially creating more false positives. Choose thresholds using a separate validation dataset and the operational cost of false positives vs. false negatives. Do not tune against the held-out test set.
