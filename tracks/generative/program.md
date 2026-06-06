# program.md — Track 3: BioMistral (QLoRA → generate the progressing cue)

You edit **`train.py`** only; `prepare.py` (prompt format + splits + fixed eval) is OFF
LIMITS. **Requires the RTX 3090** (4-bit QLoRA). This is the capstone / generative model.

## Task
Given a participant's VAAMR state + utterance, GENERATE the therapist cue most likely to
progress them across the VAAMR arc (the "operationalise the working therapist moves" goal,
ROADMAP Phase 6.3). Base: `BioMistral/BioMistral-7B`, QLoRA NF4 4-bit.

## Data — synthetic augmentation is REQUIRED here
Real 'advanced' cue blocks at n≈32 number only ~10² — far below the ~1–5k/class the
ROADMAP cites. So **generate synthetic examples first**:
```
export OPENROUTER_API_KEY=...
python -m synth.generate --data-dir ./data --n 2000 --model anthropic/claude-opus-4.8
python -m synth.validate --data-dir ./data        # filter + ablate (does it help held-out REAL ppl?)
```
Synthetic rows are quarantined to TRAIN; val/test are REAL advancers only.

## Metric (this track: lower loss is better)
- `eval_loss` = held-out token NLL on real advancing cues; `eval_ppl = exp(eval_loss)`.
- `primary_metric = -eval_loss` (higher better, uniform with other tracks).
- **Keep** iff `eval_loss` strictly decreases. `test_eval_loss` is the honest held-out number.
- Qualitative held-out generations are printed each run for inspection (not used to select).

## Budget
`TIME_BUDGET = 1800s` (30 min) — QLoRA needs more steps than BERT. Tune down for fast
iteration, but keep it fixed within a comparison campaign.

## Loop
```
LOOP:
  edit train.py (one idea) -> git commit
  uv run python tracks/generative/train.py --data-dir ./data > run.log 2>&1
  grep -E "^primary_metric:|^eval_ppl:|^test_eval_loss:|^peak_vram_mb:" run.log
  append results.tsv ; keep iff eval_loss decreased (else git reset --hard HEAD~1)
```

## Exploration dimensions (24 GB budget — watch `peak_vram_mb`)
- **LoRA**: rank {8,16,32,64}, alpha {16,32,64}, dropout, target-module set (attn-only vs +MLP).
- **Optim**: lr {1e-4..3e-4}, cosine vs linear, warmup, weight decay, max_grad_norm.
- **Throughput/mem**: `MICRO_BATCH`×`GRAD_ACCUM`, `MAX_SEQ_LEN` (prepare-fixed at 512 — if
  you need shorter, filter long examples in train, don't edit prepare), gradient checkpointing.
- **Data mix**: real:synthetic ratio (down-weight or cap synthetic), curriculum (real first),
  include `stayed` as neutral targets, condition on dominant PURER move in the prompt.
- **NEFTune** noise, packing, label-smoothing, DoRA/rsLoRA variants (if peft supports).
- **Base A/B**: `BioMistral/BioMistral-7B-DARE`, `mistralai/Mistral-7B-Instruct-v0.2`,
  `epfl-llm/meditron-7b` — change `BASE_MODEL` and the matching tokenizer in prepare's call.

## Hard rules
- Never evaluate on synthetic data. Keep the QLoRA config inside 24 GB (OOM = discard).
- Generations are research artifacts; any clinical use needs prospective validation + safety
  review (carry the caveat banner — it prints automatically).
