# Streamlit application guide

## Start the application

After installing dependencies:

```bash
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Open it in a browser.

You can start the app before training. In that case, the sidebar shows that the models are not trained. The optional **demo rules** toggle allows a limited preview of the interface. Every such result is explicitly marked `DEMO RULES — NOT A TRAINED MODEL`.

## Tabs

### Single message

Paste an SMS, email, comment, or post. Select Spam, Hate speech, or Both. The page shows the predicted class, confidence, positive-class score, and both class probabilities. Use the sidebar threshold sliders to explore decision-boundary behavior.

### Batch CSV

Upload a CSV with a column named `text`. Additional columns are retained. The output adds per-task prediction, positive-class score, and prediction source. Download the result as a CSV.

### Scenario lab

Browse the hand-written scenarios in `data/sample_cases.csv`. Filter them by scenario group, run either classifier or both, and export the results. Expected outcomes are discussion prompts, not guaranteed model outputs; ambiguous examples are intentionally included.

### Guide & model status

Check whether each fine-tuned model is present and review its saved test metrics. The page also describes the data sources and limitations.

## CSV format

Minimum input:

```csv
text
"Your parcel is arriving tomorrow."
"URGENT: claim your free reward at https://example.com"
```

To evaluate an independently labeled dataset, keep your true label in another column (for example `label`) and compare exported predictions with it using a separate evaluation workflow. Do not upload a test set used during training as if it were independent validation data.

## Troubleshooting

- **Model not trained:** run the training command for that task. Demo mode is not a substitute.
- **Out of memory:** lower `--batch-size` to 4 or 2; try fewer epochs; close other applications; use CPU if necessary.
- **Slow CPU training:** this is expected for transformer fine-tuning. A supported GPU will help.
- **Model download fails:** check the internet connection and Hugging Face access. Retry after the connection is restored.
- **CSV error:** ensure the file is a valid UTF-8 CSV and has a `text` column.
- **Wrong-language input:** the models are trained primarily on English data. Predictions on other languages should be treated as unsupported.
