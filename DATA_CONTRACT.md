# DATA_CONTRACT.md — How QRA should present training datasets to the fine-tuning workshop

This file is the interface specification between **QRA**
(`Qualitative_Research_Algorithm`, the labelling pipeline) and **AutoResearch**
(this repo, the fine-tuning workshop). It documents:

1. **What QRA emits today** — the exact files and field-level schemas the workshop
   consumes (verified against QRA `src/process/assembly/`).
2. **How each of the three models maps onto those files.**
3. **Requested additions** — concrete, prioritized changes that would make QRA's
   exports materially better training corpora. *These are documentation only — no
   PR is opened against QRA. The workshop is written to run on what QRA emits today
   and to light up the extras automatically if/when QRA adds them.*

> **Provenance note.** Everything in §1 is read straight from QRA source
> (`assembly/master_dataset.py`, `assembly/training_export.py`,
> `assembly/mindfulbert_dataset.py`). §3 is advisory.

---

## 0. Where the data lives

QRA writes training exports to `<output_dir>/02_meta/training_data/`:

| File | Producer (QRA) | Consumed by (workshop) |
|---|---|---|
| `master_segments.csv` | `assembly/master_dataset.py` | classification (fallback), all tracks (provenance) |
| `theme_classification.jsonl` | `assembly/training_export.py` | **classification** (primary) |
| `codebook_multilabel.jsonl` | `assembly/training_export.py` | classification (optional aux multi-task) |
| `label_map.json` | `assembly/training_export.py` | all tracks (label names) |
| `mindfulbert_dataset.jsonl` | `assembly/mindfulbert_dataset.py` | **nsp**, **generative** (primary) |
| `mindfulbert_datasheet.{json,txt}` | `assembly/mindfulbert_dataset.py` | all tracks (volume, provenance mix, gate, caveats) |

Pull them into the workshop with `python scripts/pull_qra_data.py --qra-output <dir>`
(copies the six files into `./data/`). The workshop never reads QRA's SQLite store
(`qra.db`) directly — only these exported artifacts.

---

## 1. Current schemas (what the workshop relies on)

### 1.1 `theme_classification.jsonl` — VAAMR classification corpus (Model 1)

One JSON object per **participant** segment that has a final VAAMR label:

```json
{
  "text": "I kept bringing my attention back to the breath, just staying with it.",
  "label": "attention regulation",     // framework short_name, lower-cased
  "label_id": 2,                         // 0..4  (the training target)
  "label_confidence_tier": "high",       // high | medium | low
  "confidence": 0.86,                    // llm_confidence_primary (0..1, may be null)
  "consistency": 3,                      // llm_run_consistency (# agreeing runs)
  "label_source": "llm_zero_shot",       // adjudicated|human_consensus|gnn_consensus|llm_zero_shot
  "segment_id": "S03_sess4_017",
  "participant_id": "P03",
  "session_id": "S03_sess4",
  "session_number": 4
}
```

**VAAMR label space (5-class, `label_map.json["theme_labels"]`):**

| `label_id` | `short_name` | stage |
|---|---|---|
| 0 | Vigilance | pre-mindfulness: attentional capture by pain |
| 1 | Avoidance | pre-mindfulness: attention used to escape experience |
| 2 | Attention Regulation | mindfulness: stable volitional presence |
| 3 | Metacognition | mindfulness: reflexive observation of mind |
| 4 | Reappraisal | mindfulness: transformation of pain's meaning |

> ⚠️ **The 4-class trap.** Earlier MORE write-ups used a 4-stage "VA-MR" (no
> *Attention Regulation*). QRA is now **5-class**. The workshop is hard-pinned to 5
> classes in `common/frameworks.py`; do not regress to 4.

### 1.2 `master_segments.csv` — the full segment table (fallback + provenance)

Generated export (the analysis layer reads it). Superset of `theme_classification.jsonl`;
includes **both speakers**. Columns the workshop uses
(`assembly/master_dataset.py:assemble_master_dataset`):

`segment_id, trial_id, participant_id, session_id, session_number, cohort_id,
session_variant, segment_index, start_time_ms, end_time_ms, speaker, text,
word_count, primary_stage, secondary_stage, llm_confidence_primary,
llm_run_consistency, purer_primary, purer_secondary, purer_final, purer_final_source,
gnn_vaamr_pred, gnn_vaamr_conf, gnn_vaamr_abstain, human_label, adjudicated_label,
final_label, final_label_source, label_confidence_tier`

Filter for the classification fallback: `speaker == "participant" AND final_label
not null`. **`progression_coord` is NOT a column here today** — see §3.2.

### 1.3 `mindfulbert_dataset.jsonl` — cue-block corpus (Models 2 & 3)

The end-goal artifact. One object per **mediated cue block**
(`FROM participant → therapist cue → TO participant`). Verified schema from
`assembly/mindfulbert_dataset.py:_build_examples`:

```json
{
  "cue_block_id": "S03_sess4_016__S03_sess4_018",
  "session_id": "S03_sess4",
  "participant_id": "P03",
  "session_number": 4,
  "context_text": "It still really hurts when I move my leg.",   // FROM participant
  "cue_text": "And as you notice that, what happens if you bring a gentle curiosity to the edge of that sensation?",  // therapist cue (concatenated turns)
  "from_stage": 0,                      // VAAMR stage BEFORE the cue (0..4)
  "to_stage": 2,                        // VAAMR stage AFTER the cue (0..4)
  "dominant_purer": 0,                  // 0..4 PURER move, or null
  "dominant_purer_name": "Phenomenology",
  "n_therapist_segments": 2,
  "n_cue_words": 17,
  "delta_progression": 2.0,             // PRIMARY regression target (signed)
  "direction": "advanced",              // PRIMARY class target: advanced|stayed|regressed
  "label_basis": "stage_difference",    // or "progression_coord" when coords exist
  "provenance": {
    "tier": "llm_zero_shot",            // WEAKEST of the two endpoints
    "from_label_source": "llm_zero_shot",
    "to_label_source": "llm_zero_shot",
    "gnn_abstain": false,
    "gate_passed": false
  },
  "augmentation": {                     // OPTIONAL, only if gate passed + retained
    "provenance": "gnn_counterfactual",
    "would_progress": 0.31
  }
}
```

**PURER move space (`dominant_purer`):** 0 Phenomenology · 1 Utilization ·
2 Reframing · 3 Education/Expectancy · 4 Reinforcement.

**Direction deadband:** `|delta_progression| ≤ 0.15` → `"stayed"`; `> 0.15` →
`"advanced"`; `< -0.15` → `"regressed"` (matches `analysis/mechanism.py`).

### 1.4 `mindfulbert_datasheet.json` — volume + provenance + gate

```json
{
  "dataset_version": "1.0", "n_examples": 412, "n_participants": 16, "n_sessions": 84,
  "label_basis": "stage_difference",
  "direction_distribution": {"advanced": 121, "stayed": 205, "regressed": 86},
  "provenance_mix": {"llm_zero_shot": 380, "human_consensus": 32},
  "n_abstained": 0, "gate_passed": false,
  "augmentation": {"enabled": false, "n_augmented": 0, "retained": false, "ablation": {}}
}
```

The workshop reads this **before training** to set class weights, decide whether the
generative track needs synthetic augmentation (it almost always will — see §2.3), and
print the n≈32 / observational / non-causal caveats into every run log.

---

## 2. Model ↔ data mapping

### 2.1 Model 1 — MindfulBERT-Classification (ClinicalBERT → VAAMR 5-class)
- **Input** `text` → **target** `label_id ∈ {0..4}`.
- **Source** `theme_classification.jsonl` (fallback: `master_segments.csv`).
- **Splits** grouped by `participant_id` (never split a participant across train/eval;
  prefer grouping by `session_id`). See `common/splits.py`.
- **Provenance** use `label_source` + `label_confidence_tier` for curriculum/weighting
  (adjudicated > human_consensus > gnn_consensus > llm_zero_shot).

### 2.2 Model 2 — MindfulBERT-NSP (BioBERT → "which phrase progresses?")
- **Input** sentence A = `context_text`, sentence B = `cue_text` → **target** binary
  *does B progress the participant?* Positive ⇔ `direction == "advanced"`.
- **Negatives** the in-block non-advancers (`direction ∈ {stayed, regressed}`) are *hard*
  negatives; mismatched (A_i, B_j) pairings are *easy* negatives (mined in
  `tracks/nsp/prepare.py`).
- **Inference / deliverable** rank a candidate pool of `cue_text`s for a given
  `context_text` by P(progress) → "the therapeutic phrase most likely to progress."
- **Source** `mindfulbert_dataset.jsonl`. Splits grouped by `participant_id`.

### 2.3 Model 3 — BioMistral (QLoRA SFT → generate the progressing cue)
- **Input** prompt = (`from_stage` name + `context_text`) → **target** = `cue_text`,
  restricted to `direction == "advanced"` (optionally `stayed`→neutral).
- **Source** `mindfulbert_dataset.jsonl` **+ synthetic** (`synth/`). With n≈32 there are
  typically only ~10²  real "advanced" cue blocks — far below the ~1–5k/class the
  ROADMAP (Phase 2.1) cites for stable fine-tuning — so Claude-Opus synthetic
  augmentation is **on by default** for this track, provenance-tagged and held-out-validated.

---

## 3. Requested additions to QRA (advisory — no PR)

Prioritised. Each is **backward-compatible** (additive). The workshop auto-detects and
uses each one if present, and degrades gracefully if absent.

### P0 — Frozen, leakage-safe split manifest  →  `02_meta/training_data/splits.json`
**Why.** Every model here and QRA's own GNN reliability gate must use *the same*
participant/session-grouped folds, or "held-out" numbers are not comparable and a
participant can leak across train/eval. The workshop computes grouped folds itself, but
a **frozen** canonical split owned by QRA removes divergence and makes the thesis numbers
reproducible.
**Proposed schema:**
```json
{"strategy": "grouped_by_participant", "k": 5, "seed": 42,
 "assignment": {"P03": 0, "P07": 1, "...": 0},
 "holdout_participants": ["P15", "P16"]}
```

### P1 — Always populate the continuous progression coordinate
**Why.** `delta_progression` falls back to integer `to_stage - from_stage`
(`label_basis="stage_difference"`) whenever `progression_coord` is missing — which is
*always* unless the full GNN ran. The continuous E[stage] coordinate (already computed by
`gnn_layer/soft_labels.py`) is a far better regression target and yields a meaningful
deadband. **Ask:** emit `progression_coord` as a column in `master_segments.csv` and as
`from_coord`/`to_coord` on every cue block, computed from the multi-run ballot mixture
**even when the GNN layer is off** (the soft-label mixture does not require the trained graph).

### P1 — Soft stage-mixture vectors on segments and cue-block endpoints
**Why.** Argmax stages throw away the superposition the methodology is built on. For
calibrated classification (soft-label distillation) and for honest Δ, expose the 5-dim
mixture. **Ask:** add `stage_mixture: [p0..p4]` to `theme_classification.jsonl`, and
`from_stage_mixture` / `to_stage_mixture` to each cue block (source: `soft_labels.py`).

### P2 — Endpoint confidences + a content hash on cue blocks
**Why.** The workshop weights examples by confidence, not just tier, and must dedupe
synthetic data against real data to prevent contamination. **Ask:** add
`from_confidence` / `to_confidence` (the `llm_confidence_primary` of each endpoint) and
`text_sha` (sha256 of `context_text|cue_text`) to each cue block.

### P2 — A deduplicated therapist-cue pool  →  `therapist_cue_pool.jsonl`
**Why.** The NSP ranker and hard-negative mining need a clean candidate set; QRA owns the
canonical cue-block builder (`process/cue_blocks.py`) and can dedupe better than the
workshop. **Proposed row:** `{"cue_text", "dominant_purer", "n_uses", "p_advance",
"text_sha"}`. (The workshop derives this itself today in `tracks/nsp/prepare.py`.)

### P3 — A ready-made SFT view  →  `mindfulbert_sft.jsonl`
**Why.** Optional convenience: QRA could emit the instruction-formatted, advancers-only
generative corpus directly. **Proposed row:**
`{"instruction", "input"(from_stage+context), "output"(cue_text), "provenance_tier",
"from_stage", "dominant_purer", "text_sha"}`. The workshop builds this from §1.3 today, so
this is lowest priority.

### P3 — Per-stage counts + class weights in the datasheet
**Why.** Saves a scan and documents imbalance. **Ask:** add `theme_label_counts:
{0:.., 1:.., ...}` to `mindfulbert_datasheet.json` and to a sibling
`theme_classification_datasheet.json`.

---

## 4. Hard rules the workshop enforces (so QRA's guarantees survive training)

1. **Never train on un-gated GNN labels.** Examples whose `provenance.tier ==
   "gnn_consensus"` are only admitted when `gate_passed == true`; the `augmentation`
   channel is dropped unless `gate_passed && retained` (mirrors QRA Decision D10).
2. **Never split a participant across train/eval.** Grouped folds only.
3. **Synthetic data is quarantined.** Tagged `provenance.tier = "synthetic_claude_opus"`,
   excluded from every eval/test split, deduped by `text_sha`, and ablated (does it help
   held-out *real* performance?) before it is kept.
4. **Carry the caveats.** Every run log reprints the datasheet's n≈32 / single-arm /
   observational / elicitation-confound caveats. The models are research artifacts; any
   clinical use needs its own prospective validation (ROADMAP Phase 6.3).
