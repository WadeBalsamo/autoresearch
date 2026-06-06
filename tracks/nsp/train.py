"""tracks/nsp/train.py — MUTABLE baseline for Model 2.

MindfulBERT-NSP: BioBERT next-sentence head fine-tuned so progress_score(context, cue)
ranks the therapist phrase most likely to advance a participant across VAAMR stages.

THIS is the file the agent edits. prepare.py (pairs + fixed eval) is OFF LIMITS.

Run:  uv run python tracks/nsp/train.py --data-dir ./data > run.log 2>&1
Keep: roc_auc strictly improves (pair discrimination is the stable selection signal).
Deliverable headline: mrr / recall@k (rank the progressing phrase first).
Metric line the loop greps:  ^primary_metric:
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForNextSentencePrediction

from prepare import (  # FIXED
    BASE_TOKENIZER, TIME_BUDGET, evaluate_nsp, evaluate_ranking, setup_data,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common import budget as B    # noqa: E402
from common import data as qdata   # noqa: E402
from common import metrics as M    # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these)
# ---------------------------------------------------------------------------
BASE_MODEL = "dmis-lab/biobert-base-cased-v1.1"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
MAX_EPOCHS = 40
GRAD_ACCUM = 1
POS_WEIGHT = 1.0     # up-weight the (rarer) progressing pairs in the loss


def _loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train(data_dir: str):
    device = B.device()
    B.reset_peak_vram()
    print(qdata.caveat_banner(data_dir))

    d = setup_data(data_dir, tokenizer_name=BASE_TOKENIZER)
    mk = d["make_dataset"]
    train_loader = _loader(mk(d["train_pairs"]), BATCH_SIZE, True)
    val_loader = _loader(mk(d["val_pairs"]), BATCH_SIZE, False)
    test_loader = _loader(mk(d["test_pairs"]), BATCH_SIZE, False)
    val_rank_loader = _loader(mk(d["val_rank"]), BATCH_SIZE, False)
    test_rank_loader = _loader(mk(d["test_rank"]), BATCH_SIZE, False)

    model = BertForNextSentencePrediction.from_pretrained(BASE_MODEL).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M  device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    # NSP label 0 == progress (the positive). Up-weight it via class weights on [isNext, notNext].
    weight = torch.tensor([POS_WEIGHT, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    total_steps = max(1, len(train_loader) * MAX_EPOCHS // GRAD_ACCUM)
    warmup = int(total_steps * WARMUP_RATIO)

    def lr_at(step):
        if step < warmup:
            return LEARNING_RATE * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return LEARNING_RATE * max(0.0, 1.0 - prog)

    bud = B.Budget(TIME_BUDGET).start()
    step = 0
    best_auc, best_state, best_pack = -1.0, None, None

    for epoch in range(MAX_EPOCHS):
        if bud.expired:
            break
        model.train()
        for batch in train_loader:
            if bud.expired:
                break
            inputs = {k: batch[k].to(device) for k in batch
                      if k in ("input_ids", "attention_mask", "token_type_ids")}
            logits = model(**inputs).logits
            loss = criterion(logits, batch["nsp_label"].to(device)) / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(step // GRAD_ACCUM)
                optimizer.step()
                optimizer.zero_grad()
            step += 1

        pair = evaluate_nsp(model, d["val_pairs"], val_loader, device)
        rank = evaluate_ranking(model, d["val_rank"], val_rank_loader, device)
        auc = pair.get("roc_auc", float("nan"))
        print(f"epoch {epoch+1}: val_roc_auc={auc:.4f} mrr={rank.get('mrr', float('nan')):.4f} "
              f"recall@5={rank.get('recall@5', float('nan')):.4f} t={bud.elapsed:.0f}s")
        if auc == auc and auc > best_auc:   # not-nan and improved
            best_auc = auc
            best_pack = {**{f"val_{k}": v for k, v in pair.items()},
                         **{f"val_{k}": v for k, v in rank.items()}}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # held-out test from best checkpoint
    test_pack = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        tp = evaluate_nsp(model, d["test_pairs"], test_loader, device)
        tr = evaluate_ranking(model, d["test_rank"], test_rank_loader, device)
        test_pack = {"test_roc_auc": tp.get("roc_auc"), "test_mrr": tr.get("mrr"),
                     "test_recall@5": tr.get("recall@5")}

    final = dict(best_pack or {})
    final.update(test_pack)
    final["primary_metric"] = best_auc if best_auc >= 0 else float("nan")
    print()
    print(M.fmt_metrics(final))
    print(f"{'training_seconds:':24s}{bud.elapsed:.1f}")
    print(f"{'peak_vram_mb:':24s}{B.peak_vram_mb():.1f}")
    print(f"{'num_params_M:':24s}{n_params/1e6:.1f}")
    print(f"{'num_steps:':24s}{step}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    train(ap.parse_args().data_dir)
