"""Torch-free loaders for QRA's training exports (see DATA_CONTRACT.md).

Everything returns plain pandas DataFrames / dicts so splits, metrics and tests run
without the DL stack. Tokenisation and tensorisation happen later, in each track's
``prepare.py``.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import pandas as pd

from . import frameworks as fw

# Provenance trust ordering (higher = more trusted). Synthetic data sits below the LLM
# floor and is quarantined from every eval split (see DATA_CONTRACT.md §4).
PROVENANCE_RANK = {
    "adjudicated": 4,
    "human_consensus": 3,
    "gnn_consensus": 2,
    "llm_zero_shot": 1,
    "synthetic_claude_opus": 0,
}
SYNTHETIC_TIER = "synthetic_claude_opus"

# Default per-tier loss weights (curriculum / down-weighting). Override via config.
TIER_WEIGHT = {
    "adjudicated": 1.0,
    "human_consensus": 1.0,
    "gnn_consensus": 0.7,
    "llm_zero_shot": 0.6,
    "synthetic_claude_opus": 0.4,
}


# --------------------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------------------
FILES = {
    "theme": "theme_classification.jsonl",
    "codebook": "codebook_multilabel.jsonl",
    "label_map": "label_map.json",
    "cue_blocks": "mindfulbert_dataset.jsonl",
    "datasheet": "mindfulbert_datasheet.json",
    "master": "master_segments.csv",
    "splits": "splits.json",                 # proposed (P0) — optional
    "cue_pool": "therapist_cue_pool.jsonl",  # proposed (P2) — optional
    "synth": "synthetic_cue_blocks.jsonl",   # produced by synth/ (workshop side)
}


def resolve(data_dir: str, key: str) -> Optional[str]:
    """Return an existing path for a known file key, else None."""
    p = os.path.join(data_dir, FILES[key])
    return p if os.path.isfile(p) else None


def text_sha(*parts: str) -> str:
    h = hashlib.sha256()
    h.update("␟".join((p or "") for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------------------
# Model 1 — classification corpus
# --------------------------------------------------------------------------------------
def classification_frame(data_dir: str) -> pd.DataFrame:
    """Unified VAAMR classification frame.

    Prefers ``theme_classification.jsonl``; falls back to participant rows of
    ``master_segments.csv``. Returns columns:
        text, label_id, label_source, label_confidence_tier, confidence,
        participant_id, session_id, session_number, segment_id
    """
    theme = resolve(data_dir, "theme")
    if theme:
        df = pd.DataFrame(_read_jsonl(theme))
        df = df.rename(columns={"label_id": "label_id"})
        df["label_id"] = df["label_id"].astype(int)
        if "label_source" not in df:
            df["label_source"] = "llm_zero_shot"
        keep = ["text", "label_id", "label_source", "label_confidence_tier",
                "confidence", "participant_id", "session_id", "session_number",
                "segment_id"]
        for c in keep:
            if c not in df:
                df[c] = None
        df = df[keep]
    else:
        master = resolve(data_dir, "master")
        if not master:
            raise FileNotFoundError(
                f"No classification corpus in {data_dir} "
                f"(need {FILES['theme']} or {FILES['master']})")
        m = pd.read_csv(master)
        m = m[(m["speaker"] == "participant") & (m["final_label"].notna())].copy()
        m["label_id"] = m["final_label"].astype(int)
        df = pd.DataFrame({
            "text": m["text"].astype(str),
            "label_id": m["label_id"],
            "label_source": m.get("final_label_source", "llm_zero_shot"),
            "label_confidence_tier": m.get("label_confidence_tier"),
            "confidence": m.get("llm_confidence_primary"),
            "participant_id": m.get("participant_id"),
            "session_id": m.get("session_id"),
            "session_number": m.get("session_number"),
            "segment_id": m.get("segment_id"),
        })

    df = df[df["text"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    df["label_source"] = df["label_source"].fillna("llm_zero_shot")
    # group key for leakage-safe splits — prefer session, fall back to participant
    df["group"] = df["participant_id"].astype(str)
    return df


# --------------------------------------------------------------------------------------
# Models 2 & 3 — cue-block corpus
# --------------------------------------------------------------------------------------
def load_cue_blocks(data_dir: str, include_synth: bool = False) -> pd.DataFrame:
    """Flatten ``mindfulbert_dataset.jsonl`` (+ optional synthetic) into a DataFrame.

    Nested ``provenance`` / ``augmentation`` are flattened to columns. A ``text_sha`` is
    computed if QRA did not provide one (proposed P2). Synthetic rows (if requested) are
    appended with ``provenance_tier == 'synthetic_claude_opus'`` and ``is_synth == True``.
    """
    path = resolve(data_dir, "cue_blocks")
    rows = _read_jsonl(path) if path else []
    recs = [_flatten_cue_block(r, is_synth=False) for r in rows]

    if include_synth:
        spath = resolve(data_dir, "synth")
        if spath:
            for r in _read_jsonl(spath):
                recs.append(_flatten_cue_block(r, is_synth=True))

    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["group"] = df["participant_id"].astype(str)
    return df


def _flatten_cue_block(r: dict, is_synth: bool) -> dict:
    prov = r.get("provenance") or {}
    aug = r.get("augmentation") or {}
    tier = prov.get("tier") or ("synthetic_claude_opus" if is_synth else "llm_zero_shot")
    if is_synth:
        tier = SYNTHETIC_TIER
    sha = r.get("text_sha") or text_sha(r.get("context_text", ""), r.get("cue_text", ""))
    return {
        "cue_block_id": r.get("cue_block_id"),
        "session_id": r.get("session_id"),
        "participant_id": r.get("participant_id"),
        "session_number": r.get("session_number"),
        "context_text": r.get("context_text", "") or "",
        "cue_text": r.get("cue_text", "") or "",
        "from_stage": r.get("from_stage"),
        "to_stage": r.get("to_stage"),
        "dominant_purer": r.get("dominant_purer"),
        "dominant_purer_name": r.get("dominant_purer_name"),
        "n_therapist_segments": r.get("n_therapist_segments"),
        "n_cue_words": r.get("n_cue_words"),
        "delta_progression": r.get("delta_progression"),
        "direction": r.get("direction"),
        "label_basis": r.get("label_basis"),
        # flattened provenance
        "provenance_tier": tier,
        "from_label_source": prov.get("from_label_source"),
        "to_label_source": prov.get("to_label_source"),
        "gnn_abstain": bool(prov.get("gnn_abstain", False)),
        "gate_passed": bool(prov.get("gate_passed", False)),
        # optional augmentation channel
        "aug_would_progress": aug.get("would_progress"),
        "aug_provenance": aug.get("provenance"),
        # proposed (P1/P2) extras — used if present, else None
        "from_coord": r.get("from_coord"),
        "to_coord": r.get("to_coord"),
        "from_confidence": r.get("from_confidence"),
        "to_confidence": r.get("to_confidence"),
        "text_sha": sha,
        "is_synth": bool(is_synth),
    }


# --------------------------------------------------------------------------------------
# provenance gating (DATA_CONTRACT §4, hard rule 1)
# --------------------------------------------------------------------------------------
def admissible(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that violate the never-train-on-ungated-GNN rule.

    A ``gnn_consensus``-tier row is admitted only when ``gate_passed``. (Synthetic rows
    are admissible for *training* but are filtered out of eval by the splitter.)
    """
    if df.empty:
        return df
    bad = (df["provenance_tier"] == "gnn_consensus") & (~df["gate_passed"].astype(bool))
    return df[~bad].reset_index(drop=True)


def tier_weights(df: pd.DataFrame, overrides: Optional[Dict[str, float]] = None) -> "pd.Series":
    w = {**TIER_WEIGHT, **(overrides or {})}
    return df["provenance_tier"].map(lambda t: w.get(t, 0.5)).astype(float)


# --------------------------------------------------------------------------------------
# datasheet / label map
# --------------------------------------------------------------------------------------
def load_datasheet(data_dir: str) -> Dict:
    p = resolve(data_dir, "datasheet")
    if not p:
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_map(data_dir: str) -> Dict:
    p = resolve(data_dir, "label_map")
    if not p:
        return {"theme_labels": {str(k): v for k, v in fw.VAAMR_STAGES.items()}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def caveat_banner(data_dir: str) -> str:
    """The n≈32 / observational caveats, reprinted into every run log (hard rule 4)."""
    ds = load_datasheet(data_dir)
    n = ds.get("n_examples", "?")
    parts = ds.get("provenance_mix", {})
    return (
        "=" * 78 + "\n"
        "CAVEATS (carried from QRA datasheet — these models are RESEARCH artifacts)\n"
        "  • n≈32 participants, single-arm, unblinded, observational.\n"
        "  • Labels are ASSOCIATIONAL, not causal (PURER inquiry elicits the very\n"
        "    language VAAMR scores — the elicitation confound is not removed here).\n"
        f"  • cue-block examples: {n}   provenance mix: {parts}\n"
        "  • Any clinical use needs its own prospective validation (ROADMAP Phase 6.3).\n"
        + "=" * 78
    )
