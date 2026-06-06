"""tracks/generative/prepare.py — FIXED data prep + eval for Model 3.

BioMistral-7B (QLoRA SFT): given a participant's VAAMR state + utterance, GENERATE the
therapist cue most likely to progress them. Trained on real 'advanced' cue blocks PLUS
Claude-Opus synthetic examples (real data is far too small at n≈32 — see DATA_CONTRACT §2.3).

⚠️ DO NOT MODIFY THIS FILE. It owns the prompt format, the leakage-safe splits and the
fixed eval (held-out response perplexity on REAL advancers only — synthetic is train-only).

Fixed eval metric:
    eval_loss = mean token NLL over the held-out real advancing cue (lower = better)
    eval_ppl  = exp(eval_loss)
    primary_metric = -eval_loss   (higher = better, uniform with the other tracks)
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import data as qdata       # noqa: E402
from common import frameworks as fw    # noqa: E402
from common import splits as S         # noqa: E402

MAX_SEQ_LEN = 512
TIME_BUDGET = 1800            # 30 min — QLoRA needs more steps than BERT to show signal
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42
BASE_TOKENIZER = "BioMistral/BioMistral-7B"

INSTRUCTION = (
    "You are a mindfulness therapist in the MORE program for chronic pain. The participant "
    "is currently expressing the '{stage}' stage of contemplative development. Reply with a "
    "single brief therapeutic cue (a guided-inquiry move) most likely to help them progress "
    "toward greater mindfulness of their experience."
)


def build_prompt(from_stage, context_text: str) -> str:
    stage = fw.vaamr_name(from_stage) if from_stage is not None else "unknown"
    return (INSTRUCTION.format(stage=stage)
            + f"\n\nParticipant: {context_text}\n\nTherapist:")


def build_sft_examples(df, advancers_only: bool = True) -> List[dict]:
    """One SFT example per progressing cue block: (prompt -> cue_text)."""
    ex = []
    for _, r in df.iterrows():
        if advancers_only and r["direction"] != "advanced":
            continue
        ctx = str(r["context_text"]).strip()
        cue = str(r["cue_text"]).strip()
        if not ctx or not cue:
            continue
        ex.append({"prompt": build_prompt(r["from_stage"], ctx), "response": cue,
                   "is_synth": bool(r.get("is_synth", False)),
                   "tier": r.get("provenance_tier")})
    return ex


class SFTDataset(Dataset):
    """Causal-LM SFT with the prompt tokens masked to -100 (loss only on the response)."""

    def __init__(self, examples: List[dict], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.ex = examples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        prompt = f"[INST] {e['prompt']} [/INST]"
        full = f"{prompt} {e['response']}{self.tok.eos_token}"
        p_ids = self.tok(prompt, add_special_tokens=True)["input_ids"]
        f_enc = self.tok(full, add_special_tokens=True, max_length=self.max_len,
                         truncation=True)
        input_ids = f_enc["input_ids"]
        labels = list(input_ids)
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = -100          # mask the prompt
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(f_enc["attention_mask"]),
                "labels": torch.tensor(labels)}


def collate(batch, pad_id: int):
    maxlen = max(len(b["input_ids"]) for b in batch)
    def pad(x, val):
        return torch.stack([torch.cat([b[x], torch.full((maxlen - len(b[x]),), val,
                                                         dtype=b[x].dtype)]) for b in batch])
    return {"input_ids": pad("input_ids", pad_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100)}


def setup_data(data_dir: str, tokenizer_name: str = BASE_TOKENIZER) -> Dict:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    df = qdata.admissible(qdata.load_cue_blocks(data_dir, include_synth=True))
    if df.empty:
        raise FileNotFoundError(f"No cue blocks in {data_dir} (need mindfulbert_dataset.jsonl)")
    train_df, val_df, test_df = S.grouped_train_val_test(
        df, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=RANDOM_SEED)
    print(S.split_summary(train_df, val_df, test_df))

    train_ex = build_sft_examples(train_df)                  # real advancers + synthetic
    val_ex = build_sft_examples(val_df[~val_df["is_synth"].astype(bool)]) if "is_synth" in val_df else build_sft_examples(val_df)
    test_ex = build_sft_examples(test_df[~test_df["is_synth"].astype(bool)]) if "is_synth" in test_df else build_sft_examples(test_df)
    n_synth = sum(1 for e in train_ex if e["is_synth"])
    print(f"SFT examples: train={len(train_ex)} (synthetic={n_synth}, real={len(train_ex)-n_synth}) "
          f"val={len(val_ex)} test={len(test_ex)}")
    if len(train_ex) - n_synth < 30:
        print("WARNING: <30 REAL advancing examples — synthetic augmentation is essential "
              "for this track (run `python -m synth.generate`).")
    return {"tokenizer": tok, "train_ex": train_ex, "val_ex": val_ex, "test_ex": test_ex,
            "SFTDataset": SFTDataset, "collate": collate}


@torch.no_grad()
def eval_loss(model, loader, device="cuda") -> Dict[str, float]:
    """FIXED metric: token-weighted mean NLL over held-out response tokens."""
    model.eval()
    total_nll, total_tok = 0.0, 0
    for batch in loader:
        out = model(input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
        n_tok = int((batch["labels"] != -100).sum())
        total_nll += float(out.loss) * n_tok
        total_tok += n_tok
    if total_tok == 0:
        return {"eval_loss": float("nan"), "eval_ppl": float("nan"), "primary_metric": float("nan")}
    mean_nll = total_nll / total_tok
    return {"eval_loss": mean_nll, "eval_ppl": math.exp(min(20.0, mean_nll)),
            "primary_metric": -mean_nll}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    a = ap.parse_args()
    print(qdata.caveat_banner(a.data_dir))
    d = setup_data(a.data_dir)
    print("Generative SFT data ready.")
