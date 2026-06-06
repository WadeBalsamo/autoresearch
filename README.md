# autoresearch — the MindfulBERT / BioMistral fine-tuning workshop

A single-GPU (RTX 3090) **fine-tuning workshop** that finds the *optimal* fine-tune of three
models on the outputs of the **Qualitative Research Algorithm** (QRA), the computational
phenomenology pipeline over the Move-MORE mindfulness-for-chronic-pain trial. This repo is a
fork/evolution of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):
keep the *autonomous, single-file, fixed-budget* experiment loop — point it at a real fine-
tuning task and let it iterate.

> **Status / caveats.** The training corpus is tiny (n≈32 participants, single-arm,
> observational). These models are **research artifacts**, not clinical tools — every run
> reprints the QRA caveats. See `DATA_CONTRACT.md`.

## The three models

| # | Model | Base | Task | Data |
|---|---|---|---|---|
| 1 | **MindfulBERT-Classification** | ClinicalBERT | participant text → VAAMR stage (**5-class**) | `theme_classification.jsonl` |
| 2 | **MindfulBERT-NSP** | BioBERT (NSP head) | (participant context, therapist phrase) → *does it progress them?* → **rank** phrases | `mindfulbert_dataset.jsonl` |
| 3 | **BioMistral** | BioMistral-7B (QLoRA) | participant state → **generate** the progressing therapist cue | `mindfulbert_dataset.jsonl` + **synthetic** |

VAAMR stages: `0` Vigilance · `1` Avoidance · `2` Attention Regulation · `3` Metacognition ·
`4` Reappraisal. PURER moves: `0` Phenomenology · `1` Utilization · `2` Reframing ·
`3` Education · `4` Reinforcement.

## Layout

```
common/        torch-free core: QRA loaders, leakage-safe splits, metrics, frameworks   (unit-tested)
tracks/
  classification/  prepare.py (FIXED) · train.py (agent edits) · program.md · results.template.tsv
  nsp/             prepare.py (FIXED) · train.py (agent edits) · program.md · results.template.tsv
  generative/      prepare.py (FIXED) · train.py (agent edits) · program.md · results.template.tsv
synth/         OpenRouter (Claude Opus) synthetic cue-block generator + validator
optimize/      hands-off hyperparameter sweep + leaderboard (complements the agent loop)
scripts/       pull_qra_data.py — copy QRA exports into ./data
tests/         torch-free unit tests + tiny QRA-format fixtures
DATA_CONTRACT.md   what QRA emits + how the workshop consumes it + requested QRA additions
program.md     top-level workshop orchestration (drive the three tracks)
```

Each track keeps Karpathy's contract: **`prepare.py` is fixed** (owns data + the fixed
metric + leakage-safe splits); **`train.py` is the one file you (or the agent) edit**;
**`program.md`** is the per-track instruction set; results log to `results.tsv`.

## Quick start (on the RTX 3090 box)

```bash
# 0. install (uv recommended)
uv sync --extra gpu           # CPU-only? drop --extra gpu to run common/synth/optimize/tests
#   (adjust the torch CUDA index in pyproject.toml to match your driver, then `uv lock`)

# 1. pull QRA's training exports into ./data
python scripts/pull_qra_data.py --qra-output /path/to/QRA/data/output --dest ./data

# 2. sanity-check the data contract (torch-free)
python -m pytest -q

# 3. classification — smoke then train
uv run python tracks/classification/prepare.py --data-dir ./data
uv run python tracks/classification/train.py --data-dir ./data > run.log 2>&1
grep -E "^macro_f1:|^test_macro_f1:|^peak_vram_mb:" run.log

# 4. NSP
uv run python tracks/nsp/train.py --data-dir ./data > run.log 2>&1

# 5. generative — generate synthetic data first (real 'advanced' blocks are scarce), then QLoRA
export OPENROUTER_API_KEY=...
python -m synth.generate --data-dir ./data --n 2000 --model anthropic/claude-opus-4.8 \
    --qra-repo /path/to/Qualitative_Research_Algorithm
python -m synth.validate --data-dir ./data
uv run python tracks/generative/train.py --data-dir ./data > run.log 2>&1
```

## Two ways to optimize

1. **Autonomous agent loop (Karpathy-style).** Point Claude/Codex at a track's `program.md`;
   it edits `train.py`, runs a fixed-budget experiment, greps `primary_metric`, keeps/discards,
   logs `results.tsv`, repeats. ~12 experiments/hour for the BERT tracks.
2. **Hands-off sweep.** `python -m optimize.search --track classification --data-dir ./data
   --trials 20` patches the editable constants, runs each under the fixed budget, and logs a
   leaderboard. `python -m optimize.leaderboard` renders `leaderboard.md`.

Both use the **same fixed splits and metric**, so results are directly comparable. The fixed
time budget means the workshop finds the best model *for your 3090 in that budget*.

## Why these design choices
- **5-class VAAMR**, always (QRA dropped the obsolete 4-stage "VA-MR" — this repo will not
  regress to it).
- **Participant-grouped splits**: at n≈32, random splits leak; we never split a participant.
- **Provenance-aware**: labels carry tiers (adjudicated > human > gnn > llm); never train on
  un-gated GNN labels; synthetic data is quarantined to train and ablated before it's trusted.
- **QLoRA for BioMistral**: 4-bit NF4 + paged 8-bit AdamW + gradient checkpointing fit a 7B
  model + SFT inside 24 GB.

See `DATA_CONTRACT.md` for the exact schemas and the additions QRA should make to present
even better training data. License: MIT (inherited from autoresearch).
