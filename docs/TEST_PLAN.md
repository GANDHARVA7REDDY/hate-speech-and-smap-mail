# Test plan and scenario coverage

## Automated checks

Run:

```bash
python -m pytest -q
```

The unit tests cover whitespace normalization, label mapping, duplicate/empty row cleanup, basic demo-mode behavior, threshold handling, and label definitions.

## Manual UI tests

| ID | Test | Expected UI behavior |
|---|---|---|
| UI-01 | Start app before training | App loads; each model is marked not trained |
| UI-02 | Submit empty text | App asks for a message; no prediction is attempted |
| UI-03 | Submit one ordinary message | Both classifiers return a result if trained, or clearly marked demo output |
| UI-04 | Choose spam only | Only the spam classifier runs |
| UI-05 | Choose hate only | Only the hate classifier runs |
| UI-06 | Change threshold | Re-run uses the new threshold and can change class decision |
| UI-07 | Upload CSV with `text` | Preview is displayed; batch output can be downloaded |
| UI-08 | Upload CSV without `text` | Validation error explains required column |
| UI-09 | Run scenario suite | Predictions and scores appear; CSV export works |
| UI-10 | Disable demo when model missing | App reports that model must be trained |

## Model-quality test categories

Use `data/sample_cases.csv` and a separately labeled in-domain evaluation set to inspect:

1. **Ordinary language:** personal scheduling, work requests, routine service alerts.
2. **Obvious spam:** unsolicited prizes, urgent claims, suspicious URLs, credential and OTP requests.
3. **Legitimate marketing:** promotional content with context and expected consent.
4. **Identity mention without hostility:** nationality, religion, gender, or other identity terms in neutral discussion.
5. **Direct group hostility:** sweeping hostile claims or exclusion directed at an identity group.
6. **Counter-speech:** quoting or rejecting a hateful statement.
7. **Reporting and education:** describing harmful language without endorsing it.
8. **Profanity and personal criticism:** rude language not necessarily targeting a protected group.
9. **Ambiguity and implicit meaning:** pronouns, coded language, sarcasm, or missing conversation context.
10. **Domain shift:** non-English text, long comments, new slang, URLs, spelling variation, and text from other platforms.

## Error analysis

Do not only report accuracy. Inspect:

- Confusion matrix and class-wise precision/recall/F1.
- False negatives: spam or hateful items predicted as negative.
- False positives: normal messages or counter-speech predicted as positive.
- Performance by message length, dialect, topic, identity group, and source domain where lawful and ethically appropriate.
- Threshold sensitivity on a separate validation set.

The sample scenarios are a sanity-check suite, not a statistically representative benchmark. A model may reasonably disagree with some ambiguous expected outcomes. Human review is necessary for consequential decisions.
