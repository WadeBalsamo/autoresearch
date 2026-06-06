import numpy as np
import pandas as pd

from common import data, splits
from conftest import FIXTURES


def _no_group_overlap(a, b):
    return set(a["group"]).isdisjoint(set(b["group"]))


def test_grouped_three_way_no_leakage():
    df = data.classification_frame(FIXTURES)
    tr, va, te = splits.grouped_train_val_test(df, val_frac=0.2, test_frac=0.2, seed=1)
    assert len(tr) + len(va) + len(te) == len(df)
    assert _no_group_overlap(tr, va)
    assert _no_group_overlap(tr, te)
    assert _no_group_overlap(va, te)
    # all three non-empty with 6 groups
    assert len(va) > 0 and len(te) > 0


def test_synthetic_rows_quarantined_to_train():
    df = data.classification_frame(FIXTURES).copy()
    df["is_synth"] = False
    # inject synthetic rows for a brand-new participant
    synth = df.head(5).copy()
    synth["is_synth"] = True
    synth["group"] = "SYNTH"
    df2 = pd.concat([df, synth], ignore_index=True)
    tr, va, te = splits.grouped_train_val_test(df2, val_frac=0.2, test_frac=0.2)
    assert "SYNTH" not in set(va["group"]) and "SYNTH" not in set(te["group"])
    assert (tr["is_synth"].sum()) == 5  # all synth landed in train


def test_grouped_kfold_disjoint():
    df = data.classification_frame(FIXTURES)
    folds = splits.grouped_kfold(df, k=3)
    assert len(folds) == 3
    for tr_idx, va_idx in folds:
        tr_groups = set(df.iloc[tr_idx]["group"])
        va_groups = set(df.iloc[va_idx]["group"])
        assert tr_groups.isdisjoint(va_groups)


def test_degrades_with_few_groups():
    df = data.classification_frame(FIXTURES)
    one = df[df["group"] == df["group"].iloc[0]]
    tr, va, te = splits.grouped_train_val_test(one, val_frac=0.2, test_frac=0.2)
    # single group -> everything in train, never leaks
    assert len(va) == 0 and len(te) == 0 and len(tr) == len(one)
