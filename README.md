# AI Spam & Hate Speech Detection

A complete educational NLP project with **two independently fine-tuned DistilBERT classifiers** and a Streamlit web app.

- **Spam classifier:** `ham` vs. `spam` using the UCI SMS Spam Collection.
- **Hate speech classifier:** `non-hate` vs. `hate` using CardiffNLP TweetEval's `hate` subset.
- **Web app:** single-message inference, batch CSV scoring, threshold controls, and a scenario-testing lab.
- **Training:** reproducible train/validation/test preparation, class-weighted fine-tuning, and evaluation reports.

> The ZIP includes source code and documentation, not third-party datasets or trained transformer weights. The first training run downloads the datasets and the pretrained base model. The app can be opened before training in clearly marked demo mode.

## 1. Project structure

```text
spam_hate_detection_ai/
├── app.py                       # Streamlit web application
├── requirements.txt
├── src/
│   ├── config.py                # Paths, model names, labels
│   ├── data.py                  # Dataset loading and preparation
│   ├── metrics.py               # Training metrics
│   ├── train.py                 # Fine-tuning entry point
│   ├── predict.py               # Local model inference
│   ├── evaluate.py              # Re-evaluation on held-out test data
│   └── demo.py                  # Clearly marked, non-AI demo fallback
├── data/
│   ├── README.md
│   └── sample_cases.csv         # Manual scenario suite
├── artifacts/                   # Fine-tuned models and metrics (created at runtime)
├── tests/                       # Unit tests
└── docs/                        # Dataset, training, UI, test, ethics, project report
```

## 2. Requirements

- Python 3.10–3.12 recommended.
- Internet connection for first-time dataset and model downloads.
- 8 GB RAM is a practical starting point; more is better.
- CPU works, but GPU training is faster. A smaller batch size may be necessary on limited hardware.

Install PyTorch appropriate for your operating system using the [official selector](https://pytorch.org/get-started/locally/) when necessary.

### Create an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train / fine-tune the models

Train both:

```bash
python -m src.train --task all --epochs 3 --batch-size 8
```

Train individually:

```bash
python -m src.train --task spam --epochs 3 --batch-size 8
python -m src.train --task hate --epochs 3 --batch-size 8
```

For a quick smoke run, try `--epochs 1`. It is not a production-quality training run. The scripts download the data and `distilbert-base-uncased` automatically on first use. Training outputs are saved in `artifacts/spam_model/` and `artifacts/hate_model/`.

## 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`).

### App capabilities

1. **Single message:** choose spam, hate speech, or both; inspect class probabilities and confidence.
2. **Batch CSV:** upload a CSV with a `text` column, score every row, and download the results.
3. **Scenario lab:** run the included examples covering ordinary messages, prize/OTP scams, neutral identity mentions, group-directed hostility, counterspeech, quotation, profanity, sarcasm, ambiguity, and non-English input.
4. **Guide & model status:** see which models are trained and inspect saved test metrics.

When model files are missing, the optional demo fallback previews the interface using simple hand-written rules. It is always marked as **DEMO RULES — NOT A TRAINED MODEL**. Disable the fallback to require real trained models.

## 5. Evaluate models

Training automatically saves a held-out test report. Re-run evaluation with:

```bash
python -m src.evaluate --task spam --threshold 0.50
python -m src.evaluate --task hate --threshold 0.50
```

Review precision, recall, F1, macro-F1, and the confusion matrix. Do not rely on accuracy alone, especially for imbalanced spam data. Tune thresholds on validation data, not on the final test set.

## 6. Run tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## 7. Dataset definitions and citations

**Spam:** UCI SMS Spam Collection, 5,574 English messages labeled ham/spam. Original source: [UCI dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection); Hugging Face loader: [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam). Citation: Almeida et al. (2011), [DOI 10.24432/C5CC84](https://doi.org/10.24432/C5CC84).

**Hate speech:** [`cardiffnlp/tweet_eval`, `hate` configuration](https://huggingface.co/datasets/cardiffnlp/tweet_eval), with labels `0=non-hate`, `1=hate`. See the [TweetEval project](https://github.com/cardiffnlp/TweetEval). The dataset card currently lists the license as unknown; review the source terms before redistribution or deployment.

More detail: [docs/DATASETS.md](docs/DATASETS.md).

## 8. Documentation

- [Project report and architecture](docs/PROJECT_REPORT.md)
- [Dataset definitions and usage notes](docs/DATASETS.md)
- [Fine-tuning procedure](docs/MODEL_TRAINING.md)
- [Streamlit user guide](docs/USER_GUIDE.md)
- [Test plan and scenarios](docs/TEST_PLAN.md)
- [Ethics and limitations](docs/ETHICS_AND_LIMITATIONS.md)

## 9. Important limitations

This is an educational English-language prototype, not a production moderation service. Dataset labels are imperfect; context, sarcasm, counterspeech, quotation, dialect, and domain shift can cause errors. A model confidence score is not proof of malicious intent. Do not automatically punish or restrict people based only on these predictions. Use qualified human review, privacy safeguards, and a representative validation dataset for any consequential use.
