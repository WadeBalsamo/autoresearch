"""scripts/pull_qra_data.py — copy QRA's training exports into the workshop's ./data dir.

QRA writes to ``<qra_output>/02_meta/training_data/``. This copies the six contract files
(see DATA_CONTRACT.md §0) plus the optional proposed extras if QRA emits them. It never
reads QRA's SQLite store directly.

Usage:
    python scripts/pull_qra_data.py --qra-output /path/to/qra/data/output --dest ./data
"""
from __future__ import annotations

import argparse
import os
import shutil

CONTRACT = [
    "master_segments.csv", "theme_classification.jsonl", "codebook_multilabel.jsonl",
    "label_map.json", "mindfulbert_dataset.jsonl",
    "mindfulbert_datasheet.json", "mindfulbert_datasheet.txt",
]
OPTIONAL = ["splits.json", "therapist_cue_pool.jsonl",
            "theme_classification_datasheet.json", "mindfulbert_sft.jsonl"]


def _candidates(qra_output: str):
    yield os.path.join(qra_output, "02_meta", "training_data")
    yield os.path.join(qra_output, "training_data")
    yield qra_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qra-output", required=True, help="QRA output_dir (or its training_data dir)")
    ap.add_argument("--dest", default="./data")
    args = ap.parse_args()

    src_dir = next((d for d in _candidates(args.qra_output)
                    if os.path.isfile(os.path.join(d, "mindfulbert_dataset.jsonl"))
                    or os.path.isfile(os.path.join(d, "theme_classification.jsonl"))), None)
    if src_dir is None:
        raise SystemExit(f"Could not find QRA training exports under {args.qra_output} "
                         "(looked for 02_meta/training_data/). Has the pipeline run?")
    os.makedirs(args.dest, exist_ok=True)

    copied, missing = [], []
    for name in CONTRACT:
        s = os.path.join(src_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dest, name)); copied.append(name)
        else:
            missing.append(name)
    for name in OPTIONAL:
        s = os.path.join(src_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dest, name)); copied.append(name + " (optional)")

    print(f"source: {src_dir}\ndest:   {os.path.abspath(args.dest)}")
    print("copied:\n  " + "\n  ".join(copied) if copied else "copied: nothing")
    if missing:
        print("MISSING (workshop degrades gracefully, but check QRA ran the relevant stage):")
        for m in missing:
            print(f"  - {m}")
    if any(o.startswith("splits.json") for o in copied):
        print("note: QRA splits.json present — the workshop will honour the frozen folds.")
    else:
        print("note: no splits.json — the workshop computes participant-grouped folds itself "
              "(see DATA_CONTRACT.md P0).")


if __name__ == "__main__":
    main()
