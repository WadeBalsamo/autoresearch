"""tracks/classification/train.py — MUTABLE baseline for Model 1.

MindfulBERT-Classification: ClinicalBERT -> VAAMR 5-class stage classifier.

THIS is the file the autoresearch agent edits. Everything below the hyperparameter block
is fair game: head architecture, pooling, layer freezing, layer-wise LR decay, focal /
label-smoothed / curriculum losses, provenance-weighted sampling, augmentation, etc.
prepare.py (data + fixed eval) is OFF LIMITS.

Run:  uv run python tracks/classification/train.py --data-dir ./data > run.log 2>&1
Keep: macro_f1 strictly improves AND every per-class F1 >= MIN_PER_CLASS_F1 (0.20).
Metric line the loop greps:  ^macro_f1:
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from prepare import (  # FIXED imports — do not reach around these
    BASE_TOKENIZER, MIN_PER_CLASS_F1, NUM_CLASSES, TIME_BUDGET,
    evaluate_classification, print_evaluation_results, setup_data,
)

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.dirname(__file__))))
from common import budget as B   # noqa: E402
from common import data as qdata  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these)
# ---------------------------------------------------------------------------
BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
DROPOUT = 0.1
WARMUP_RATIO = 0.1
MAX_EPOCHS = 50            # capped by TIME_BUDGET anyway
GRAD_ACCUM = 1
LABEL_SMOOTHING = 0.0
USE_PROVENANCE_WEIGHTS = False   # multiply CE by per-example tier weight (curriculum)


class MindfulBERTClassifier(nn.Module):
    """ClinicalBERT encoder + a classification head on [CLS]. Baseline = single linear."""

    def __init__(self, model_name=BASE_MODEL, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_classes)

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))


def train(data_dir: str):
    device = B.device()
    B.reset_peak_vram()
    print(qdata.caveat_banner(data_dir))

    d = setup_data(data_dir, tokenizer_name=BASE_TOKENIZER)
    train_loader = DataLoader(d["train_dataset"], batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(d["val_dataset"], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(d["test_dataset"], batch_size=BATCH_SIZE, shuffle=False)

    model = MindfulBERTClassifier().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M  device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    class_w = d["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING,
                                    reduction="none")

    total_steps = max(1, len(train_loader) * MAX_EPOCHS // GRAD_ACCUM)
    warmup = int(total_steps * WARMUP_RATIO)

    def lr_at(step):
        if step < warmup:
            return LEARNING_RATE * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return LEARNING_RATE * max(0.0, 1.0 - prog)

    bud = B.Budget(TIME_BUDGET).start()
    step = 0
    best_macro, best_val = 0.0, None
    best_state = None

    for epoch in range(MAX_EPOCHS):
        if bud.expired:
            break
        model.train()
        for batch in train_loader:
            if bud.expired:
                break
            logits = model(batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device))
            per_ex = criterion(logits, batch["label"].to(device))
            if USE_PROVENANCE_WEIGHTS:
                per_ex = per_ex * batch["weight"].to(device)
            loss = per_ex.mean() / GRAD_ACCUM
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(step // GRAD_ACCUM)
                optimizer.step()
                optimizer.zero_grad()
            step += 1

        val = evaluate_classification(model, val_loader, device=device)
        print(f"epoch {epoch+1}: val_macro_f1={val['macro_f1']:.4f} "
              f"kappa={val['kappa']:.4f} min_class_f1={val['min_per_class_f1']:.3f} "
              f"t={bud.elapsed:.0f}s")
        # keep best val checkpoint that respects the per-class floor
        if val["macro_f1"] > best_macro and val["min_per_class_f1"] >= MIN_PER_CLASS_F1:
            best_macro, best_val = val["macro_f1"], val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # honest held-out test number from the best val checkpoint
    test = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        test = evaluate_classification(model, test_loader, device=device)

    final = dict(best_val or evaluate_classification(model, val_loader, device=device))
    if test:
        final["test_macro_f1"] = test["macro_f1"]
        final["test_kappa"] = test["kappa"]
    print()
    print_evaluation_results(final)
    print(f"{'training_seconds:':24s}{bud.elapsed:.1f}")
    print(f"{'peak_vram_mb:':24s}{B.peak_vram_mb():.1f}")
    print(f"{'num_params_M:':24s}{n_params/1e6:.1f}")
    print(f"{'num_steps:':24s}{step}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    train(ap.parse_args().data_dir)
