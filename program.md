# program.md — workshop orchestration (drive all three tracks)

This is the top-level instruction set for an autonomous research agent running the
MindfulBERT / BioMistral fine-tuning workshop. Each track has its own detailed
`program.md`; this file says how to sequence them and the rules that span all three.

## Mission
Find the **optimal fine-tune** of three models on QRA's outputs, on a single RTX 3090:
1. `tracks/classification` — ClinicalBERT → VAAMR 5-class (`program.md`)
2. `tracks/nsp` — BioBERT NSP → "which therapist phrase progresses the participant?" (`program.md`)
3. `tracks/generative` — BioMistral-7B QLoRA → generate the progressing cue (`program.md`)

## Order of operations
1. **Data.** `python scripts/pull_qra_data.py --qra-output <QRA_OUT> --dest ./data`, then
   `python -m pytest -q` (torch-free contract/splits/metrics must pass), then eyeball
   `data/mindfulbert_datasheet.json` for volume + provenance mix.
2. **Classification first.** Cheapest, fastest signal; its best checkpoint also serves as the
   synthetic-data content-validity checker (`synth/validate.py --classifier`).
3. **NSP second.** Shares the cue-block data; establishes the progression-scoring baseline.
4. **Synthetic data.** If `data/mindfulbert_datasheet.json` shows < ~200 'advanced' blocks
   (it will), run `python -m synth.generate` (Claude Opus) then `python -m synth.validate`.
5. **Generative last.** QLoRA SFT with real + validated synthetic; confirm synthetic *helps*
   by comparing held-out REAL `test_eval_loss` with vs without it.

## Per-track loop (same shape for all three)
```
LOOP (per track):
  1. cp tracks/<t>/results.template.tsv tracks/<t>/results.tsv   # once
  2. read git state + tail last run.log
  3. edit ONLY tracks/<t>/train.py with one idea (prepare.py is OFF LIMITS)
  4. git commit -m "<t>: <idea>"
  5. uv run python tracks/<t>/train.py --data-dir ./data > run.log 2>&1
  6. grep -E "^primary_metric:|^peak_vram_mb:" run.log    # classification also emits ^macro_f1:
  7. on crash: tail -n 50 run.log, fix, retry
  8. append a row to tracks/<t>/results.tsv
  9. KEEP (commit stays) iff the track's keep-criterion holds (see its program.md); else
     git reset --hard HEAD~1
 10. never stop until the budget/leaderboard plateaus
```

Or run the hands-off optimizer: `python -m optimize.search --track <t> --data-dir ./data
--trials N`, then `python -m optimize.leaderboard`.

## Rules that span all tracks (from DATA_CONTRACT.md §4)
- **5-class VAAMR**, always. Never the 4-stage VA-MR.
- **Never split a participant across train/eval.** Use the provided grouped splits only.
- **Never train on un-gated GNN labels.** `common.data.admissible()` already drops them.
- **Synthetic is train-only**, deduped, ablated. It must lower held-out REAL loss to stay.
- **Carry the caveats.** They print automatically; do not strip them. No clinical claims.
- **Don't touch `prepare.py`, the splits, or the eval** — comparability depends on them.

## What "done" looks like
A `leaderboard.md` with, per track: the best config, its held-out test metric, peak VRAM,
and a one-line provenance/caveat note — i.e. the optimal 3090 fine-tune of each model, with
an audit trail of what was tried.
