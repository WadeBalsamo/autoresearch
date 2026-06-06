"""Metrics for all three tracks — pure numpy/sklearn, no torch.

Classification (Model 1): macro-F1 (primary), Cohen's κ, per-class F1 (+ floor).
NSP ranking (Model 2): ROC-AUC / AP on the pair task, plus MRR & recall@k for the
"which phrase progresses?" retrieval framing. Calibration: ECE.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------------------
# classification (Model 1)
# --------------------------------------------------------------------------------------
def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           num_classes: int = 5) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "min_per_class_f1": float(np.min(per)) if len(per) else 0.0,
    }
    for i in range(num_classes):
        out[f"f1_class{i}"] = float(per[i]) if i < len(per) else 0.0
    return out


def confusion(y_true, y_pred, num_classes: int = 5) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


# --------------------------------------------------------------------------------------
# NSP / progression ranking (Model 2)
# --------------------------------------------------------------------------------------
def pair_metrics(y_true: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """Binary pair task: y=1 means 'this cue progresses the participant'."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    out = {"n_pairs": int(len(y_true)), "pos_rate": float(y_true.mean()) if len(y_true) else 0.0}
    if len(set(y_true.tolist())) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["avg_precision"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["avg_precision"] = float("nan")
    # accuracy at the natural 0.5 threshold on a sigmoid score, if scores look like probs
    preds = (scores >= 0.5).astype(int) if scores.min() >= 0 and scores.max() <= 1 else (scores >= np.median(scores)).astype(int)
    out["pair_acc"] = float((preds == y_true).mean()) if len(y_true) else 0.0
    return out


def ranking_metrics(groups: Sequence, y_true: Sequence[int], scores: Sequence[float],
                    ks: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    """Retrieval framing: per query (group = a participant context), rank candidate cues
    by ``scores`` and ask whether the actually-progressing cues rank at the top.

    Returns MRR (first relevant hit) and recall@k, averaged over queries that have ≥1
    positive and ≥2 candidates.
    """
    groups = np.asarray(groups)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    mrr_vals: List[float] = []
    recall: Dict[int, List[float]] = {k: [] for k in ks}
    for g in np.unique(groups):
        m = groups == g
        yt, sc = y_true[m], scores[m]
        if yt.sum() == 0 or len(yt) < 2:
            continue
        order = np.argsort(-sc)
        ranked = yt[order]
        first = np.argmax(ranked == 1)
        mrr_vals.append(1.0 / (first + 1))
        n_pos = int(yt.sum())
        for k in ks:
            recall[k].append(float(ranked[:k].sum()) / n_pos)
    out = {"n_queries": len(mrr_vals),
           "mrr": float(np.mean(mrr_vals)) if mrr_vals else float("nan")}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recall[k])) if recall[k] else float("nan")
    return out


# --------------------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, y_true: Sequence[int],
                               n_bins: int = 10) -> float:
    """ECE for a multi-class softmax matrix ``probs`` (rows sum to 1)."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fmt_metrics(d: Dict[str, float]) -> str:
    """Print metrics in the grep-able ``key: value`` block the agent loop reads."""
    lines = ["---"]
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"{k + ':':24s}{v:.6f}")
        else:
            lines.append(f"{k + ':':24s}{v}")
    return "\n".join(lines)
