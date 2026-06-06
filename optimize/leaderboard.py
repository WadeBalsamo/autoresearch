"""optimize/leaderboard.py — aggregate sweep + hand-run results into leaderboard.md.

Reads ``runs/<track>/sweep.tsv`` (optimizer) and ``tracks/<track>/results.tsv`` (agent /
human loop) and renders a ranked Markdown leaderboard per track.
"""
from __future__ import annotations

import argparse
import csv
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = ["classification", "nsp", "generative"]


def _read_tsv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _topn(rows, key, higher_better=True, n=10):
    def val(r):
        try:
            return float(r.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")
    rows = [r for r in rows if val(r) == val(r)]  # drop nan
    return sorted(rows, key=val, reverse=higher_better)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "leaderboard.md"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    lines = ["# Fine-tuning workshop — leaderboard", ""]
    for t in TRACKS:
        lines.append(f"## {t}")
        sweep = _read_tsv(os.path.join(_ROOT, "runs", t, "sweep.tsv"))
        if sweep:
            lines.append(f"\n**Optimizer sweep** (top {args.n} by `primary_metric`):\n")
            lines.append("| primary_metric | trial | config |")
            lines.append("|---|---|---|")
            for r in _topn(sweep, "primary_metric", True, args.n):
                lines.append(f"| {r.get('primary_metric')} | {r.get('trial')} | "
                             f"`{r.get('config','')[:120]}` |")
        else:
            lines.append("\n_(no optimizer sweep yet — run `python -m optimize.search "
                         f"--track {t} --data-dir ./data`)_")
        human = _read_tsv(os.path.join(_ROOT, "tracks", t, "results.tsv"))
        kept = [r for r in human if str(r.get("status", "")).lower() in ("keep", "baseline")]
        if kept:
            lines.append("\n**Hand-run / agent keeps:**\n")
            for r in kept[-args.n:]:
                desc = r.get("description", "")
                pm = r.get("macro_f1") or r.get("primary_metric") or ""
                lines.append(f"- `{r.get('commit','')[:8]}` **{pm}** — {desc}")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
