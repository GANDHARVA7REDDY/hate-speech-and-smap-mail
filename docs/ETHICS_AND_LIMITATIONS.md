# Responsible use, limitations, and safety

This repository is an educational prototype. It is not a production trust-and-safety system, a legal classifier, or a substitute for trained human moderation.

## Important limitations

- **Language:** both selected datasets are English. Non-English inputs are outside the intended evaluation scope.
- **Domain shift:** SMS messages and Twitter posts differ from private chats, email, forums, and other platforms.
- **Label ambiguity:** spam and hate speech labels depend on dataset-specific annotation rules and context.
- **Context:** quotation, satire, counterspeech, reclaimed language, threats, coded language, and conversation history may change interpretation.
- **Bias:** models can learn correlations that disproportionately flag particular groups, dialects, or identity terms. A mention of an identity is not itself hate speech.
- **Confidence is not truth:** softmax confidence is not a calibrated probability of intent, harm, or policy violation.
- **Dataset age and coverage:** the SMS collection and TweetEval reflect their collection periods and may not represent current scams, slang, or social norms.
- **No identity or intent inference:** do not use outputs to infer a person's character, beliefs, identity, or criminality.

## Appropriate use

- Learning NLP, transformer fine-tuning, model evaluation, and Streamlit app development.
- Assisting a trained reviewer by surfacing potentially relevant content for review.
- Conducting controlled offline experiments with documented datasets.

## Inappropriate use

- Automatically punishing, suspending, firing, denying services, or reporting people based only on model output.
- Treating a positive score as proof of malicious intent or hate speech.
- Using the demo rule fallback for real decisions.
- Uploading confidential, private, or personal messages without appropriate authority and safeguards.

## Recommended safeguards before deployment

1. Define a narrow policy and operational meaning for each label.
2. Obtain rights to use a representative, consent-aware dataset from the intended platform and language communities.
3. Have independent reviewers annotate a validation set and measure inter-annotator disagreement.
4. Audit false positives and false negatives across relevant subgroups and dialects.
5. Calibrate thresholds on validation data and document the costs of each error type.
6. Keep a human appeal and correction mechanism.
7. Minimize stored text, restrict access, define retention, and protect uploaded files.
8. Monitor drift and re-evaluate after any retraining or policy change.

## Demo fallback

The `src/demo.py` module contains transparent keyword/regular-expression heuristics so the Streamlit UI can be explored before training. It is deliberately labeled as a demo and does not claim to understand context. It is not a baseline model and must not be compared with the fine-tuned transformer metrics.
