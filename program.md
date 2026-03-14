# autoresearch: VA-MR Classification Fine-Tuning
 
This is autoresearch for fine-tuning a classification model
(Mindful-BERT) on the Vigilance-Avoidance Metacognition-Reappraisal (VA-MR)
framework for therapeutic dialogue analysis.
 
## Domain Context
 
The VA-MR framework describes four stages of contemplative transformation
that participants express during Mindfulness-Oriented Recovery Enhancement:
 
- **Stage 0 - Vigilance**: Pain hypervigilance, attentional fragmentation,
  catastrophic thinking. Attention is reactive rather than directed.
- **Stage 1 - Avoidance**: Attentional control deployed for avoidance rather
  than investigation. Deliberately pushing pain away, distraction as strategy.
- **Stage 2 - Metacognition**: Observing one's own mental processes. Noticing
  reactions, watching thoughts arise and pass, stepping back from experience.
- **Stage 3 - Reappraisal**: Fundamental reinterpretation of sensory experience.
  Pain as changing, composed of distinct sensations, lacking fixed significance.
 
The training data consists of transcript segments from therapy sessions,
labeled with these four stages by a zero-shot LLM pipeline validated against
human qualitative coders.
 
## Setup
 
1. **Agree on a run tag** (e.g. `vamr_mar14`).
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the files**:
   - `prepare_vamr.py` — fixed data prep, evaluation function. DO NOT MODIFY.
   - `train_vamr.py` — the file you modify. Model, optimizer, training loop.
   - This file — domain context and instructions.
4. **Verify data exists**: The master segment dataset should be at the path
   specified by `--dataset`.
5. **Initialize results.tsv** with header row.
6. **Confirm and go**.
 
## Constraints
 
**What you CAN do:**
- Modify `train_vamr.py` — everything is fair game: model architecture,
  optimizer, hyperparameters, training loop, batch size, pooling strategy,
  data augmentation, loss function, layer freezing, etc.
 
**What you CANNOT do:**
- Modify `prepare_vamr.py`. It contains the fixed evaluation and data loading.
- Install new packages.
- Modify the evaluation function.
 
## Metric
 
**The goal: get the highest macro_f1.**
 
Higher is better (opposite of val_bpb in the original autoresearch).
 
The keep criterion:
1. macro_f1 must strictly improve over current best
2. **AND** no single per-class F1 may drop below 0.25
 
This compound criterion prevents abandoning minority classes. The
`label_confidence_tier` column is available for curriculum learning.
 
## Output format
 
The script prints:
```
---
macro_f1:             0.782300
kappa:                0.7156
f1_vigilance:         0.8102
f1_avoidance:         0.7234
f1_metacognition:     0.7891
f1_reappraisal:       0.7065
training_seconds:     300.1
peak_vram_mb:         4500.2
num_params_M:         110.0
num_steps:            450
```
 
Extract the key metric: `grep "^macro_f1:" run.log`
 
## Logging results
 
Log to `results.tsv` (tab-separated):
 
```
commit	macro_f1	memory_gb	status	description
a1b2c3d	0.782300	4.4	keep	baseline ClinicalBERT + linear head
b2c3d4e	0.795100	4.5	keep	MLP head (768->256->4) with ReLU
c3d4e5f	0.781000	4.4	discard	switch to mean pooling (slight decrease)
```
 
## Exploration Dimensions
 
Small-data classification regime. Key directions:
 
- **Architecture**: Base model (ClinicalBERT, BioBERT, RoBERTa, DeBERTa-v3),
  classification head (linear, MLP, attention-weighted pooling), pooling strategy
- **Training**: Layer freezing, layer-wise LR decay, label smoothing,
  focal loss, mixup augmentation, contrastive losses
- **Regularization**: Dropout, weight decay, early stopping
- **Curriculum**: Train on high-confidence segments first (use `label_confidence_tier`)
- **Multi-task**: Binary auxiliary task predicting early vs. late stages
- **Augmentation**: Synonym replacement, random token deletion, back-translation
 
## The experiment loop
 
Same as original autoresearch:
 
LOOP FOREVER:
1. Look at git state
2. Modify `train_vamr.py` with an experimental idea
3. git commit
4. Run: `uv run train_vamr.py --dataset <path> > run.log 2>&1`
5. Extract: `grep "^macro_f1:\|^peak_vram_mb:" run.log`
6. If crashed: check `tail -n 50 run.log`, try to fix
7. Log results to results.tsv
8. If macro_f1 improved AND all per-class F1 >= 0.25: keep
9. If not: git reset
10. **NEVER STOP** — run indefinitely until manually interrupted
