# Dataset documentation

## 1. Spam: UCI SMS Spam Collection

**Loaded as:** `ucirvine/sms_spam` from the Hugging Face Datasets Hub. The original collection is maintained by the UCI Machine Learning Repository.

**Original source:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)  
**Dataset citation:** Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. Proceedings of the 2011 ACM Symposium on Document Engineering. [DOI: 10.24432/C5CC84](https://doi.org/10.24432/C5CC84)  
**Hub copy:** [ucirvine/sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam)

| Field | Meaning |
|---|---|
| `sms` | Message text |
| `label` | Original class label |
| `0` | Ham / legitimate |
| `1` | Spam |

The collection contains 5,574 English messages. The commonly reported distribution is 4,827 ham and 747 spam (about 86.6% vs. 13.4%), so accuracy alone can hide poor spam recall. This project creates stratified 80% train / 10% validation / 10% test splits with a fixed seed. Exact counts can vary slightly after duplicate and empty-text removal.

The script intentionally keeps punctuation, numbers, URLs, and casing. Those can carry useful spam signals. It only collapses whitespace, removes blank rows, and removes exact duplicate texts before splitting.

## 2. Hate speech: CardiffNLP TweetEval — `hate`

**Loaded as:** `load_dataset("cardiffnlp/tweet_eval", "hate")`  
**Dataset card:** [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)  
**Paper / project:** [TweetEval — Unified Benchmark and Comparative Evaluation for Tweet Classification](https://github.com/cardiffnlp/TweetEval)

TweetEval contains several Twitter text-classification tasks. This project uses only its binary `hate` configuration and preserves its published train, validation, and test splits.

| Label ID | Meaning |
|---|---|
| `0` | Non-hate |
| `1` | Hate |

The dataset card describes English-language tweets. It is not a universal definition of hate speech and does not cover every region, language, dialect, identity group, or platform. It may contain offensive or distressing examples.

**Licensing note:** the currently surfaced Hugging Face dataset card lists the license as unknown. Check the dataset card, upstream repository, and intended use restrictions before redistributing data or deploying a model trained on it. This project downloads the dataset at runtime and does not redistribute it.

## 3. Why two datasets and two models?

Spam and hate speech are distinct concepts. A message may be spam but not hateful, hateful but not spam, both, or neither. Separate binary classifiers keep labels and evaluation interpretable and allow each model to be improved independently.

## 4. Data governance

- Do not place private messages, phone numbers, personal details, or confidential content in a public dataset or Git repository.
- Use the source datasets in accordance with their terms and cite them in derivative research.
- Avoid treating dataset labels as objective truth. Labels reflect collection and annotation guidelines.
- Keep the train/test separation intact. Do not tune thresholds on the test set.
- For a real deployment, collect a lawful, consent-aware, representative validation set from the intended domain and have qualified human reviewers audit errors.
