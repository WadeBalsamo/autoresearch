"""synth/generate.py — generate synthetic cue blocks via OpenRouter (Claude Opus).

Schema-matched to QRA's ``mindfulbert_dataset.jsonl`` so the tracks consume it unchanged.
Every row is tagged ``provenance.tier = 'synthetic_claude_opus'`` (quarantined to TRAIN,
deduped against real data by ``text_sha``). Appends as it goes (checkpointing), so an
interrupted run keeps what it produced.

Usage:
    export OPENROUTER_API_KEY=...
    python -m synth.generate --data-dir ./data --n 2000 \
        --model anthropic/claude-opus-4.8 --qra-repo ../Qualitative_Research_Algorithm
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import data as qdata          # noqa: E402
from common import frameworks as fw       # noqa: E402
from synth import prompts as P            # noqa: E402
from synth.openrouter_client import OpenRouterClient, DEFAULT_MODEL  # noqa: E402


def _existing_shas(data_dir: str, out_path: str) -> set:
    shas = set()
    real = qdata.resolve(data_dir, "cue_blocks")
    if real:
        df = qdata.load_cue_blocks(data_dir)
        shas |= set(df["text_sha"].tolist())
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    shas.add(json.loads(line).get("text_sha"))
                except Exception:
                    pass
    return shas


def _to_row(ex: dict, idx: int) -> "dict | None":
    try:
        fs, ts = int(ex["from_stage"]), int(ex["to_stage"])
        mv = int(ex["dominant_purer"])
        ctx = str(ex["context_text"]).strip()
        cue = str(ex["cue_text"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if not ctx or not cue:
        return None
    if not (0 <= fs <= 4 and 0 <= ts <= 4 and 0 <= mv <= 4):
        return None
    delta = float(ts - fs)
    direction = ("advanced" if delta > fw.PROGRESS_DEADBAND
                 else "regressed" if delta < -fw.PROGRESS_DEADBAND else "stayed")
    sha = qdata.text_sha(ctx, cue)
    return {
        "cue_block_id": f"synth_{idx:06d}",
        "session_id": "synthetic", "participant_id": f"SYNTH_{idx % 64:02d}",
        "session_number": None,
        "context_text": ctx, "cue_text": cue,
        "from_stage": fs, "to_stage": ts,
        "dominant_purer": mv, "dominant_purer_name": fw.purer_name(mv),
        "n_therapist_segments": 1, "n_cue_words": len(cue.split()),
        "delta_progression": round(delta, 3), "direction": direction,
        "label_basis": "synthetic",
        "provenance": {"tier": "synthetic_claude_opus",
                       "from_label_source": "synthetic_claude_opus",
                       "to_label_source": "synthetic_claude_opus",
                       "gnn_abstain": False, "gate_passed": False},
        "text_sha": sha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default=None, help="default: <data-dir>/synthetic_cue_blocks.jsonl")
    ap.add_argument("--n", type=int, default=2000, help="target number of synthetic examples")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--neg-fraction", type=float, default=0.0,
                    help="fraction of non-advancing (stayed/regressed) examples for NSP negatives")
    ap.add_argument("--qra-repo", default=None,
                    help="optional QRA checkout to enrich prompts with full framework defs")
    args = ap.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    out = args.out or os.path.join(args.data_dir, "synthetic_cue_blocks.jsonl")
    client = OpenRouterClient(model=args.model)

    extra = fw.load_qra_markdown(args.qra_repo) if args.qra_repo else None
    system = P.system_prompt(extra)
    seen = _existing_shas(args.data_dir, out)
    print(f"model={args.model}  target={args.n}  already-have-shas={len(seen)}  out={out}")

    written, idx, dup, bad = 0, len(seen), 0, 0
    # round-robin coverage over FROM stages 0..3 (4 is terminal) x PURER moves 0..4
    combos = [(fs, mv) for fs in range(0, 4) for mv in range(0, 5)]
    ci = 0
    with open(out, "a", encoding="utf-8") as f:
        while written < args.n:
            fs, mv = combos[ci % len(combos)]
            ci += 1
            want = "advanced"
            if args.neg_fraction > 0 and (ci % max(1, int(1 / args.neg_fraction)) == 0):
                want = "stayed"
            user = P.batch_user_prompt(args.batch_size, from_stage=fs, purer=mv,
                                       want_direction=want)
            try:
                resp = client.chat(system, user, temperature=args.temperature)
            except Exception as e:
                print(f"  request failed ({e}); continuing")
                continue
            for ex in P.parse_examples(resp):
                row = _to_row(ex, idx)
                if row is None:
                    bad += 1
                    continue
                if row["text_sha"] in seen:
                    dup += 1
                    continue
                seen.add(row["text_sha"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                idx += 1
                if written >= args.n:
                    break
            if written and written % 50 < args.batch_size:
                print(f"  written={written}/{args.n}  dup={dup} bad={bad}")
    print(f"DONE: wrote {written} synthetic examples to {out} (dup={dup}, rejected={bad})")
    print("Next: `python -m synth.validate --data-dir ./data` then train the generative track "
          "with vs without synthetic to confirm it lowers held-out REAL ppl.")


if __name__ == "__main__":
    main()
