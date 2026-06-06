"""synth/validate.py — sanity-check, dedupe, and report on synthetic cue blocks.

What it always does (CPU, no DL stack):
  • schema + range validation, direction/Δ consistency
  • duplicate / n-gram contamination check against REAL cue blocks (train/test hygiene)
  • coverage (per FROM stage, per PURER move) and lexical-diversity report
  • writes ``synthetic_datasheet.json``

Optional content-validity round-trip (needs a trained Model-1 classifier + transformers):
  --classifier <hf_dir>  → classify each synthetic ``context_text`` and report how often the
  predicted VAAMR stage matches the claimed ``from_stage``.

The DEFINITIVE "does synthetic help?" test is the generative track itself: train with vs
without synthetic and compare held-out REAL ``test_eval_loss`` (this script just gatekeeps
quality before you spend GPU time).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import data as qdata          # noqa: E402
from common import frameworks as fw       # noqa: E402


def _ngrams(text: str, n: int):
    toks = text.lower().split()
    return set(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--classifier", default=None,
                    help="optional HF dir of a trained Model-1 classifier for round-trip CV")
    args = ap.parse_args()

    synth_path = qdata.resolve(args.data_dir, "synth")
    if not synth_path:
        print(f"No synthetic file at {os.path.join(args.data_dir, qdata.FILES['synth'])}. "
              "Run `python -m synth.generate` first.")
        return
    synth = qdata.load_cue_blocks(args.data_dir, include_synth=True)
    synth = synth[synth["is_synth"]].reset_index(drop=True)
    real = qdata.load_cue_blocks(args.data_dir, include_synth=False)

    report = {"n_synth": int(len(synth)), "n_real": int(len(real))}

    # consistency: advanced ⇒ to>from
    adv = synth[synth["direction"] == "advanced"]
    ok = (adv["to_stage"].astype(float) > adv["from_stage"].astype(float)).mean() if len(adv) else 1.0
    report["advanced_to_gt_from_rate"] = round(float(ok), 4)
    report["direction_distribution"] = dict(Counter(synth["direction"]))
    report["from_stage_coverage"] = {fw.vaamr_name(s): int((synth["from_stage"] == s).sum())
                                     for s in range(5)}
    report["purer_coverage"] = {fw.purer_name(m): int((synth["dominant_purer"] == m).sum())
                               for m in range(5)}

    # contamination vs real (shared 4-gram in cue_text → potential leakage)
    real_4 = set()
    for t in real["cue_text"]:
        real_4 |= _ngrams(str(t), 4)
    contaminated = sum(1 for t in synth["cue_text"] if _ngrams(str(t), 4) & real_4)
    report["cue_4gram_overlap_with_real"] = int(contaminated)
    report["exact_dup_vs_real"] = int(synth["text_sha"].isin(set(real["text_sha"])).sum())

    # diversity
    all_cue_3 = Counter()
    for t in synth["cue_text"]:
        all_cue_3.update(_ngrams(str(t), 3))
    total_cues = max(1, len(synth))
    report["distinct_cue_3grams_per_example"] = round(len(all_cue_3) / total_cues, 3)
    report["mean_cue_words"] = round(float(synth["n_cue_words"].astype(float).mean()), 2)

    # optional round-trip content validity
    if args.classifier:
        report["content_validity"] = _roundtrip(synth, args.classifier)

    out = os.path.join(args.data_dir, "synthetic_datasheet.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    _verdict(report)


def _roundtrip(synth, classifier_dir):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        return {"status": f"skipped (transformers/torch unavailable: {e})"}
    if not os.path.isdir(classifier_dir):
        return {"status": f"skipped (no classifier dir {classifier_dir})"}
    tok = AutoTokenizer.from_pretrained(classifier_dir)
    model = AutoModelForSequenceClassification.from_pretrained(classifier_dir)
    model.eval()
    agree = 0
    for _, r in synth.iterrows():
        enc = tok(str(r["context_text"]), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            pred = int(model(**enc).logits.argmax(-1))
        agree += int(pred == int(r["from_stage"]))
    return {"status": "ok", "from_stage_agreement": round(agree / max(1, len(synth)), 4)}


def _verdict(rep):
    print("-" * 60)
    if rep["exact_dup_vs_real"] > 0:
        print(f"⚠ {rep['exact_dup_vs_real']} synthetic rows duplicate REAL data — dedup failed.")
    if rep["cue_4gram_overlap_with_real"] > 0.1 * max(1, rep["n_synth"]):
        print("⚠ high 4-gram overlap with real cues — possible memorisation/contamination.")
    if rep["advanced_to_gt_from_rate"] < 0.95:
        print("⚠ some 'advanced' rows don't have to_stage>from_stage — generator drift.")
    print("Synthetic data is TRAIN-ONLY and auto-quarantined from eval. The real test is "
          "whether it lowers the generative track's held-out REAL test_eval_loss.")


if __name__ == "__main__":
    main()
