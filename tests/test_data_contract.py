import os

import pandas as pd

from common import data, frameworks as fw
from conftest import FIXTURES


def test_classification_frame_from_theme_jsonl():
    df = data.classification_frame(FIXTURES)
    assert len(df) == 48
    assert set(df.columns) >= {"text", "label_id", "group", "participant_id", "label_source"}
    assert df["label_id"].between(0, 4).all()
    assert df["group"].nunique() == 6
    # every label is one of the 5 VAAMR stages
    assert set(df["label_id"].unique()) <= set(range(fw.VAAMR_NUM_CLASSES))


def test_classification_fallback_to_master_csv(tmp_path):
    # a data dir with ONLY master_segments.csv must still yield the participant rows
    import shutil
    shutil.copy(os.path.join(FIXTURES, "master_segments.csv"), tmp_path / "master_segments.csv")
    df = data.classification_frame(str(tmp_path))
    assert len(df) == 48  # therapist rows filtered out
    assert (df["label_id"] >= 0).all()


def test_cue_blocks_flatten_and_admissibility():
    df = data.load_cue_blocks(FIXTURES)
    assert len(df) == 18
    assert {"context_text", "cue_text", "direction", "provenance_tier",
            "gate_passed", "text_sha"} <= set(df.columns)
    # text_sha computed when QRA didn't supply it
    assert df["text_sha"].str.len().eq(16).all()
    # the single ungated gnn_consensus row is dropped (hard rule 1)
    adm = data.admissible(df)
    assert len(adm) == 17
    assert not ((adm["provenance_tier"] == "gnn_consensus") & (~adm["gate_passed"])).any()


def test_tier_weights_and_datasheet():
    df = data.load_cue_blocks(FIXTURES)
    w = data.tier_weights(df)
    assert (w > 0).all() and (w <= 1.0).all()
    ds = data.load_datasheet(FIXTURES)
    assert ds["n_examples"] == 18
    banner = data.caveat_banner(FIXTURES)
    assert "n≈32" in banner and "associational" in banner.lower()


def test_label_map_is_five_class():
    lm = data.load_label_map(FIXTURES)
    assert len(lm["theme_labels"]) == 5
    assert lm["theme_labels"]["2"].lower() == "attention regulation"
