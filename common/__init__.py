"""Shared, torch-free infrastructure for the MindfulBERT/BioMistral fine-tuning workshop.

Importing this package must NOT require torch/transformers — only numpy, pandas and
scikit-learn — so the data contract, splits, metrics and synthetic-data tooling can be
exercised and unit-tested on a CPU-only box. The heavy DL stack is imported lazily,
inside the per-track ``prepare.py`` / ``train.py`` files only.
"""

from . import frameworks, data, splits, metrics  # noqa: F401

__all__ = ["frameworks", "data", "splits", "metrics"]
