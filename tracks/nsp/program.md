# program.md — Track 2: MindfulBERT-NSP (BioBERT → "which phrase progresses?")

You edit **`train.py`** only; `prepare.py` (pairs + fixed eval + splits) is OFF LIMITS.

## Task
Reframe BioBERT's Next-Sentence-Prediction head as a **progression scorer**: given a
participant's current utterance A, score whether a therapist phrase B is the response most
likely to **advance them across VAAMR stages**. At inference, rank a candidate pool of
therapist phrases for A and return the top one. Base: `dmis-lab/biobert-base-cased-v1.1`.

Label convention: progress→NSP label 0 (isNext), not-progress→1. `progress_score =
softmax(logits)[:,0]`.

## Data
Cue blocks (`mindfulbert_dataset.jsonl`). Positives = `direction=='advanced'`; hard
negatives = stayed/regressed; easy negatives = mismatched cues from other participants
(mined in prepare). Synthetic cue blocks (if generated) are train-only and auto-quarantined.

## Metric
- **Selection (keep)**: `roc_auc` on the held-out pair task (stable with small n).
- **Deliverable headline**: `mrr`, `recall@5` — does the model rank the *actually
  progressing* cue above distractors?  Also `test_*` (honest held-out).
- Metric line the loop greps: `^primary_metric:` (== best val roc_auc).
- **Keep** iff `roc_auc` strictly improves AND `mrr` does not collapse (≥ random 1/|cands|).

## Loop
```
LOOP:
  edit train.py (one idea) -> git commit
  uv run python tracks/nsp/train.py --data-dir ./data > run.log 2>&1
  grep -E "^primary_metric:|^test_roc_auc:|^mrr:|^peak_vram_mb:" run.log
  append results.tsv ; keep iff roc_auc improved (else git reset --hard HEAD~1)
```

## Exploration dimensions
- **Negatives**: `NEG_RATIO` lives in prepare (fixed), but you control sampling temperature
  in train via reweighting; try hard-negative mining (score all mismatches, keep the
  hardest), in-batch negatives, contrastive / triplet loss on the [CLS] relationship vector.
- **Loss**: `POS_WEIGHT` for class imbalance; focal loss; margin ranking loss on
  (pos_score − neg_score).
- **Head**: NSP head as-is vs a fresh bi-encoder (encode A and B separately, cosine) vs
  cross-encoder MLP on pooled [CLS]. A bi-encoder makes the inference-time ranking over a
  large cue pool cheap.
- **Conditioning**: prepend the FROM-stage name to A (e.g. "[STAGE=Avoidance] ...") so the
  model can learn stage-moderated effects (the methodology's H2).
- **Encoder A/B**: BioBERT vs ClinicalBERT vs PubMedBERT; cased vs uncased.
- **Augmentation**: pull synthetic (context, cue, advanced) positives from `synth/` to
  fight positive scarcity — they are quarantined from eval automatically.

## Hard rules
- Don't touch prepare.py / splits / eval. Never evaluate on synthetic rows.
- Report `test_*` honestly; tiny n means MRR is noisy — watch the trend, not one run.
