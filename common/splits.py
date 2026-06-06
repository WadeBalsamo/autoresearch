"""Leakage-safe, participant/session-grouped splits (DATA_CONTRACT §4, hard rule 2).

A participant must never appear in more than one of {train, val, test}; otherwise
"held-out" performance is inflated by within-participant correlation. Synthetic rows
(``is_synth == True``) are quarantined to TRAIN only — never evaluated on (hard rule 3).
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

SEED = 42


def _split_real_synth(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "is_synth" in df.columns:
        synth = df[df["is_synth"].astype(bool)]
        real = df[~df["is_synth"].astype(bool)]
        return real.reset_index(drop=True), synth.reset_index(drop=True)
    return df.reset_index(drop=True), df.iloc[0:0]


def grouped_train_val_test(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
    group_col: str = "group",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way split with no group spanning splits. Synthetic → train only.

    Degrades gracefully when there are too few groups to honour the fractions: it will
    still never leak a group, but val/test may be smaller (or, with <3 groups, empty) —
    the caller should check and warn.
    """
    real, synth = _split_real_synth(df)
    n_groups = real[group_col].nunique()

    if n_groups < 2 or (val_frac + test_frac) <= 0:
        # cannot hold out a disjoint group set; put everything in train
        train = pd.concat([real, synth], ignore_index=True)
        empty = real.iloc[0:0]
        return train.reset_index(drop=True), empty, empty

    groups = real[group_col].to_numpy()

    # stage 1: carve out test by group
    if test_frac > 0 and n_groups >= 3:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest_idx, test_idx = next(gss.split(real, groups=groups))
        rest, test = real.iloc[rest_idx], real.iloc[test_idx]
    else:
        rest, test = real, real.iloc[0:0]

    # stage 2: carve val out of the remainder, again by group
    rest_groups = rest[group_col].to_numpy()
    if val_frac > 0 and rest[group_col].nunique() >= 2:
        adj_val = val_frac / max(1e-9, (1.0 - test_frac))
        adj_val = min(max(adj_val, 0.0), 0.9)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=adj_val, random_state=seed + 1)
        tr_idx, val_idx = next(gss2.split(rest, groups=rest_groups))
        train, val = rest.iloc[tr_idx], rest.iloc[val_idx]
    else:
        train, val = rest, rest.iloc[0:0]

    train = pd.concat([train, synth], ignore_index=True)
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def grouped_kfold(
    df: pd.DataFrame, k: int = 5, group_col: str = "group",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Group-disjoint k-fold over the REAL rows. Returns positional (train_idx, val_idx).

    Synthetic rows are appended to every fold's train indices (never to val).
    """
    real, synth = _split_real_synth(df)
    n_groups = real[group_col].nunique()
    k_eff = max(2, min(k, n_groups))
    if n_groups < 2:
        idx = np.arange(len(df))
        return [(idx, np.array([], dtype=int))]

    gkf = GroupKFold(n_splits=k_eff)
    real_pos = real.index.to_numpy()  # positions within `real`
    synth_pos = (np.arange(len(synth)) + len(real)) if len(synth) else np.array([], dtype=int)
    folds = []
    for tr, va in gkf.split(real, groups=real[group_col].to_numpy()):
        tr_all = np.concatenate([real_pos[tr], synth_pos]) if len(synth_pos) else real_pos[tr]
        folds.append((tr_all.astype(int), real_pos[va].astype(int)))
    return folds


def load_frozen_assignment(data_dir: str) -> Optional[dict]:
    """Load QRA's proposed ``splits.json`` (P0) if present."""
    p = os.path.join(data_dir, "splits.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def split_summary(train, val, test, group_col: str = "group") -> str:
    def g(d):
        return 0 if d is None or len(d) == 0 else d[group_col].nunique()
    return (f"split: train={len(train)} rows/{g(train)} groups  "
            f"val={len(val)}/{g(val)}  test={len(test)}/{g(test)}")
