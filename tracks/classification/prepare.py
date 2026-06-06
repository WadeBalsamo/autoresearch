"""tracks/classification/prepare.py — FIXED data prep + eval for Model 1.

MindfulBERT-Classification: participant segment text -> VAAMR stage (5-class).
Base encoder: ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT).

⚠️ DO NOT MODIFY THIS FILE in the autoresearch loop. It owns the fixed evaluation and
the leakage-safe (participant-grouped) splits, so experiments stay comparable. The agent
edits ``train.py`` only.

Constants (fixed):
    NUM_CLASSES = 5         (Vigilance, Avoidance, Attention Regulation, Metacognition, Reappraisal)
    MAX_SEQ_LEN = 256       (participant segments are short; 256 covers the long tail)
    TIME_BUDGET = 300       (5 min/experiment, excludes model download)
    VAL/TEST grouped by participant_id  (no participant spans splits)
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# repo root on path -> `import common`
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import data as qdata          # noqa: E402
from common import frameworks as fw        # noqa: E402
from common import metrics as M            # noqa: E402
from common import splits as S             # noqa: E402

NUM_CLASSES = fw.VAAMR_NUM_CLASSES         # 5
MAX_SEQ_LEN = 256
TIME_BUDGET = 300
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
MIN_PER_CLASS_F1 = 0.20                    # floor: keep decision rejects abandoning a class
BASE_TOKENIZER = "emilyalsentzer/Bio_ClinicalBERT"


class VAAMRDataset(Dataset):
    """Tokenises on the fly. Carries per-example provenance weight for optional curriculum."""

    def __init__(self, df, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.weights = qdata.tier_weights(
            df.assign(provenance_tier=df["label_source"])).astype(float).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=self.max_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
            "weight": torch.tensor(self.weights[i], dtype=torch.float),
        }


def class_weights(train_df) -> torch.Tensor:
    counts = train_df["label_id"].value_counts().to_dict()
    total = len(train_df)
    w = [total / (NUM_CLASSES * max(1, counts.get(i, 0))) for i in range(NUM_CLASSES)]
    return torch.tensor(w, dtype=torch.float32)


def setup_data(data_dir: str, tokenizer_name: str = BASE_TOKENIZER) -> Dict:
    """One-call setup for train.py. Returns tokenizer, datasets, frames, class weights."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    df = qdata.classification_frame(data_dir)
    train_df, val_df, test_df = S.grouped_train_val_test(
        df, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=RANDOM_SEED)
    print(S.split_summary(train_df, val_df, test_df))
    print("train label dist:", train_df["label_id"].value_counts().sort_index().to_dict())
    return {
        "tokenizer": tok,
        "train_dataset": VAAMRDataset(train_df, tok),
        "val_dataset": VAAMRDataset(val_df, tok),
        "test_dataset": VAAMRDataset(test_df, tok),
        "class_weights": class_weights(train_df),
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


@torch.no_grad()
def evaluate_classification(model, loader, device="cuda") -> Dict[str, float]:
    """FIXED metric. macro_f1 is the primary keep/discard signal (higher is better)."""
    model.eval()
    preds, labels, probs = [], [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device),
                       attention_mask=batch["attention_mask"].to(device))
        if hasattr(logits, "logits"):
            logits = logits.logits
        p = torch.softmax(logits, dim=-1)
        preds.extend(p.argmax(-1).cpu().numpy())
        probs.extend(p.cpu().numpy())
        labels.extend(batch["label"].numpy())
    res = M.classification_metrics(labels, preds, num_classes=NUM_CLASSES)
    if probs:
        res["ece"] = M.expected_calibration_error(np.array(probs), labels)
    return res


def print_evaluation_results(res: Dict[str, float]):
    print(M.fmt_metrics(res))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    a = ap.parse_args()
    print(qdata.caveat_banner(a.data_dir))
    d = setup_data(a.data_dir)
    print(f"Ready: train={len(d['train_dataset'])} val={len(d['val_dataset'])} "
          f"test={len(d['test_dataset'])}  class_weights={d['class_weights'].tolist()}")
