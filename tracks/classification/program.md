# program.md — Track 1: MindfulBERT-Classification (ClinicalBERT → VAAMR 5-class)

Autonomous fine-tuning loop. You edit **`train.py`** only; `prepare.py` (data + fixed
eval + leakage-safe splits) is OFF LIMITS.

## Task
Classify a **participant** segment into one of the **5 VAAMR stages**
(0 Vigilance · 1 Avoidance · 2 Attention Regulation · 3 Metacognition · 4 Reappraisal).
Base encoder: `emilyalsentzer/Bio_ClinicalBERT`.

## Setup (once)
1. `git checkout -b cls/<run-tag>`
2. Pull data: `python scripts/pull_qra_data.py --qra-output <QRA_OUTPUT_DIR>`  → `./data`
3. Smoke the data: `uv run python tracks/classification/prepare.py --data-dir ./data`
4. `cp tracks/classification/results.template.tsv tracks/classification/results.tsv`

## Metric (higher is better)
- Primary: **`macro_f1`** (validation, participant-grouped).
- **Keep** iff: `macro_f1` strictly improves over the best so far **AND** every
  per-class F1 ≥ `MIN_PER_CLASS_F1` (0.20). The floor stops the model from sacrificing
  the rare mindfulness stages (Metacognition/Reappraisal) for macro gains.
- Also reported: `kappa`, `ece`, `test_macro_f1` (the honest held-out number — never
  tune against it), `peak_vram_mb`.

## The loop (run forever)
```
LOOP:
  1. read git state + tail of last run.log
  2. edit train.py with ONE idea
  3. git commit -m "cls: <idea>"
  4. uv run python tracks/classification/train.py --data-dir ./data > run.log 2>&1
  5. grep -E "^macro_f1:|^test_macro_f1:|^peak_vram_mb:" run.log
  6. on crash: tail -n 50 run.log, fix, retry
  7. append a row to results.tsv
  8. keep (commit stays) iff macro_f1 improved AND all per-class F1 >= 0.20; else `git reset --hard HEAD~1`
```

## Exploration dimensions (small-data, ~10²–10³ examples/class)
- **Head**: linear → MLP (768→256→5, GELU) → attention-pooled → mean+max concat pooling.
- **Encoder**: freeze bottom-k layers; layer-wise LR decay (e.g. 0.95^depth); unfreeze schedule.
- **Loss**: class-weighted CE (on) · `LABEL_SMOOTHING` {0.0,0.05,0.1} · focal loss · CORAL/
  ordinal loss (VAAMR is an ordinal arc — exploit it).
- **Curriculum**: `USE_PROVENANCE_WEIGHTS=True` weights examples by label tier
  (adjudicated/human > llm_zero_shot); or train high-confidence first.
- **Regularisation**: dropout, weight decay, early stop on val.
- **Base model A/B**: try `medicalai/ClinicalBERT`, `dmis-lab/biobert-base-cased-v1.1`,
  `mental/mental-bert-base-uncased` (the ROADMAP "psych-bert" idea) — change `BASE_MODEL`
  AND `BASE_TOKENIZER` together (the tokenizer lives in prepare's constant; pass a matching
  `tokenizer_name` into `setup_data`).
- **Augmentation**: synonym swap, random token dropout, back-translation; or pull synthetic
  participant segments (see `synth/`) — synthetic is train-only and auto-quarantined.

## Hard rules
- 5 classes, always. Never collapse to the obsolete 4-stage VA-MR.
- Don't touch prepare.py, the splits, or the eval.
- Report honestly; the per-class floor is non-negotiable.
