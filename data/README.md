# Data directory

The project deliberately does **not** bundle full third-party datasets. The training scripts download them using Hugging Face `datasets` when first run and cache them locally.

- `sample_cases.csv` is a small, hand-written interface test suite. It is not training data and is not used for model fitting.
- Downloaded datasets are cached by Hugging Face outside this repository by default.
- Generated metrics and trained model weights are saved under `artifacts/` and are excluded from Git by `.gitignore` because they can be large.

See [`../docs/DATASETS.md`](../docs/DATASETS.md) for sources, label mappings, usage notes, and limitations.
