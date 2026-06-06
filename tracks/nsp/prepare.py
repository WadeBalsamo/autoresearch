"""tracks/nsp/prepare.py — FIXED data prep + eval for Model 2.

MindfulBERT-NSP: reframe BioBERT's Next-Sentence-Prediction head as
    "does therapist phrase B *progress* the participant who just said A?"
so that, at inference, we can rank a pool of candidate therapist phrases for a given
participant state and return the one most likely to progress them across VAAMR stages.

Base: dmis-lab/biobert-base-cased-v1.1 (BertForNextSentencePrediction).

LABEL CONVENTION (important):
    progress (advanced)      -> NSP label 0  (= "isNext": B is the progressing response)
    not-progress / mismatch  -> NSP label 1  (= "notNext")
    progress_score = softmax(seq_relationship_logits)[:, 0]

⚠️ DO NOT MODIFY THIS FILE. It owns the fixed splits, pair construction and eval.

Pairs:
  positive  : (context, its own cue)         where direction == 'advanced'      y=1
  hard neg  : (context, its own cue)         where direction in {stayed,regressed} y=0
  easy neg  : (context, a cue from ANOTHER participant)                          y=0
Ranking eval: for each 'advanced' context build [true cue] + sampled distractor cues and
ask whether the model ranks the true progressing cue at the top (MRR / recall@k).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import data as qdata      # noqa: E402
from common import metrics as M       # noqa: E402
from common import splits as S        # noqa: E402

MAX_SEQ_LEN = 256
TIME_BUDGET = 300
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
NEG_RATIO = 1.0                # easy (mismatched) negatives per positive
RANK_NUM_CANDIDATES = 10       # 1 true progressing cue + 9 distractors
BASE_TOKENIZER = "dmis-lab/biobert-base-cased-v1.1"


# --------------------------------------------------------------------------------------
# pair construction (fixed, seeded)
# --------------------------------------------------------------------------------------
def _sample_other_cues(df: pd.DataFrame, exclude_participant, n, rng) -> List[str]:
    pool = df[df["participant_id"] != exclude_participant]["cue_text"]
    if len(pool) == 0:
        pool = df["cue_text"]
    if len(pool) == 0:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=len(pool) < n)
    return pool.iloc[idx].tolist()


def make_pairs(df: pd.DataFrame, neg_ratio: float = NEG_RATIO,
               seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df.iterrows():
        y = 1 if r["direction"] == "advanced" else 0
        rows.append({"context_text": r["context_text"], "cue_text": r["cue_text"],
                     "y": y, "group": str(r["participant_id"]),
                     "kind": "pos" if y else "hard_neg"})
        if y == 1 and neg_ratio > 0:  # mine easy (mismatched) negatives for each positive
            for cue in _sample_other_cues(df, r["participant_id"], int(round(neg_ratio)), rng):
                rows.append({"context_text": r["context_text"], "cue_text": cue,
                             "y": 0, "group": str(r["participant_id"]), "kind": "easy_neg"})
    return pd.DataFrame(rows)


def build_ranking_eval(df: pd.DataFrame, num_candidates: int = RANK_NUM_CANDIDATES,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """For each 'advanced' context: 1 true cue (y=1) + distractor cues from others (y=0).
    ``group`` identifies the query so ranking_metrics can rank within it."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    pos = df[df["direction"] == "advanced"].reset_index(drop=True)
    for qi, r in pos.iterrows():
        gid = f"q{qi}"
        rows.append({"context_text": r["context_text"], "cue_text": r["cue_text"],
                     "y": 1, "group": gid})
        for cue in _sample_other_cues(df, r["participant_id"], num_candidates - 1, rng):
            rows.append({"context_text": r["context_text"], "cue_text": cue,
                         "y": 0, "group": gid})
    return pd.DataFrame(rows)


class NSPPairDataset(Dataset):
    def __init__(self, pairs: pd.DataFrame, tokenizer, max_len: int = MAX_SEQ_LEN):
        self.a = pairs["context_text"].astype(str).tolist()
        self.b = pairs["cue_text"].astype(str).tolist()
        self.y = pairs["y"].astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        enc = self.tok(self.a[i], self.b[i], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["y_progress"] = torch.tensor(self.y[i], dtype=torch.long)
        item["nsp_label"] = torch.tensor(0 if self.y[i] == 1 else 1, dtype=torch.long)
        return item


def setup_data(data_dir: str, tokenizer_name: str = BASE_TOKENIZER) -> Dict:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    df = qdata.admissible(qdata.load_cue_blocks(data_dir, include_synth=True))
    if df.empty:
        raise FileNotFoundError(f"No cue blocks in {data_dir} (need mindfulbert_dataset.jsonl)")
    train_df, val_df, test_df = S.grouped_train_val_test(
        df, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=RANDOM_SEED)
    print(S.split_summary(train_df, val_df, test_df))

    train_pairs = make_pairs(train_df)
    val_pairs = make_pairs(val_df)
    test_pairs = make_pairs(test_df)
    val_rank = build_ranking_eval(val_df)
    test_rank = build_ranking_eval(test_df)
    print(f"pairs: train={len(train_pairs)} (pos={int(train_pairs.y.sum())}) "
          f"val={len(val_pairs)} test={len(test_pairs)}  "
          f"ranking queries: val={val_rank.group.nunique()} test={test_rank.group.nunique()}")
    return {
        "tokenizer": tok,
        "train_pairs": train_pairs, "val_pairs": val_pairs, "test_pairs": test_pairs,
        "val_rank": val_rank, "test_rank": test_rank,
        "make_dataset": lambda pairs: NSPPairDataset(pairs, tok),
    }


# --------------------------------------------------------------------------------------
# fixed eval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _scores(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        inputs = {k: batch[k].to(device) for k in batch
                  if k in ("input_ids", "attention_mask", "token_type_ids")}
        logits = model(**inputs)
        logits = logits.logits if hasattr(logits, "logits") else logits
        prob_progress = torch.softmax(logits, dim=-1)[:, 0]  # NSP label 0 == progress
        out.append(prob_progress.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def evaluate_nsp(model, pairs: pd.DataFrame, loader, device="cuda") -> Dict[str, float]:
    s = _scores(model, loader, device)
    return M.pair_metrics(pairs["y"].to_numpy(), s)


def evaluate_ranking(model, rank: pd.DataFrame, loader, device="cuda") -> Dict[str, float]:
    s = _scores(model, loader, device)
    return M.ranking_metrics(rank["group"].to_numpy(), rank["y"].to_numpy(), s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    a = ap.parse_args()
    print(qdata.caveat_banner(a.data_dir))
    d = setup_data(a.data_dir)
    print("NSP data ready.")
