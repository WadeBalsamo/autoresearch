"""
prepare.py 
---------------
Fixed data preparation for VA-MR classification fine-tuning via AutoResearch.
 
This replaces the language modeling prepare.py for the classification task.
It loads the validated labeled dataset from the Qualitative Research Algorithm
pipeline and provides a fixed evaluation function for the AutoResearch
experiment loop.
 
CONSTANTS (do NOT modify):
    NUM_CLASSES = 4       (Vigilance, Avoidance, Metacognition, Reappraisal)
    MAX_SEQ_LEN = 512     (ClinicalBERT max sequence length)
    TIME_BUDGET = 300     (5 minutes per experiment)
    VAL_SPLIT   = 0.15
    TEST_SPLIT  = 0.15
    RANDOM_SEED = 42
 
Usage:
    python prepare_vamr.py --dataset /path/to/master_segments.csv
"""
 
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from transformers import AutoTokenizer
 
# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
 
NUM_CLASSES = 4
MAX_SEQ_LEN = 512
TIME_BUDGET = 300          # seconds (5 minutes)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
MIN_PER_CLASS_F1 = 0.25    # floor for per-class F1 in keep/discard decisions
 
STAGE_NAMES = ['Vigilance', 'Avoidance', 'Metacognition', 'Reappraisal']
 
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_vamr")
DATA_DIR = os.path.join(CACHE_DIR, "data")
 
# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
 
def load_labeled_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the master segment dataset produced by the VA-MR labeling pipeline.
 
    Filters to participant segments with valid final_label.
    """
    if dataset_path.endswith('.jsonl'):
        df = pd.read_json(dataset_path, lines=True)
    elif dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported format: {dataset_path}")
 
    # Filter to participant segments with labels
    df = df[
        (df['speaker'] == 'participant')
        & (df['final_label'].notna())
    ].copy()
    df['final_label'] = df['final_label'].astype(int)
 
    print(f"Loaded {len(df)} labeled participant segments")
    print(f"Label distribution:\n{df['final_label'].value_counts().sort_index()}")
 
    return df
 
 
def make_stratified_splits(
    df: pd.DataFrame,
) -> tuple:
    """
    Construct stratified train/val/test splits using trial_id and final_label.
 
    Ensures each split has proportional representation of all four stages
    AND all four trials.
    """
    # Create stratification key combining trial and label
    df['strat_key'] = df['trial_id'].astype(str) + '_' + df['final_label'].astype(str)
 
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, random_state=RANDOM_SEED,
        stratify=df['strat_key'],
    )
 
    # Second split: separate validation from training
    val_frac = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_frac, random_state=RANDOM_SEED,
        stratify=train_val_df['strat_key'],
    )
 
    df.drop(columns=['strat_key'], inplace=True)
    train_df = train_df.drop(columns=['strat_key'], errors='ignore')
    val_df = val_df.drop(columns=['strat_key'], errors='ignore')
    test_df = test_df.drop(columns=['strat_key'], errors='ignore')
 
    print(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df
 
 
def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequency.
    """
    counts = train_df['final_label'].value_counts().sort_index()
    total = len(train_df)
    weights = torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )
    print(f"Class weights: {weights.tolist()}")
    return weights
 
 
# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
 
class VAMRDataset(Dataset):
    """
    Dataset for VA-MR classification fine-tuning.
 
    Tokenizes text with ClinicalBERT tokenizer, pads/truncates to MAX_SEQ_LEN.
    """
 
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['final_label'].astype(int).tolist()
        self.confidence_tiers = df.get('label_confidence_tier', pd.Series(['medium'] * len(df))).tolist()
        self.tokenizer = tokenizer
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
 
 
# ---------------------------------------------------------------------------
# Evaluation function (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------
 
def evaluate_classification(model, val_loader, device='cuda'):
    """
    Fixed evaluation function for VA-MR classification.
 
    Returns a dict with all metrics. The primary metric for keep/discard
    decisions is macro_f1.
 
    The agent reads macro_f1 the same way autoresearch reads val_bpb --
    just now higher is better instead of lower.
    """
    model.eval()
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
 
            logits = model(input_ids, attention_mask=attention_mask)
            if hasattr(logits, 'logits'):
                logits = logits.logits
 
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
 
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
 
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
 
    results = {
        'macro_f1': round(float(macro_f1), 6),
        'kappa': round(float(kappa), 4),
    }
    for i, name in enumerate(STAGE_NAMES):
        results[f'f1_{name.lower()}'] = round(float(per_class_f1[i]), 4)
 
    return results
 
 
def print_evaluation_results(results: dict):
    """Print evaluation results in the format that the agent greps for."""
    print("---")
    for key, val in results.items():
        print(f"{key + ':':22s}{val}")
 
 
# ---------------------------------------------------------------------------
# Content validity evaluation
# ---------------------------------------------------------------------------
 
def evaluate_content_validity(
    model, tokenizer, test_set_path: str, device='cuda',
) -> dict:
    """
    Evaluate content validity on the prototypical test set.
 
    Loads the content_validity_test_set.jsonl and checks whether
    the model correctly classifies prototypical expressions.
    """
    if not os.path.exists(test_set_path):
        print(f"Warning: content validity test set not found at {test_set_path}")
        return {}
 
    model.eval()
    items = []
    with open(test_set_path) as f:
        for line in f:
            items.append(json.loads(line))
 
    correct_by_stage = {i: {'correct': 0, 'total': 0} for i in range(NUM_CLASSES)}
 
    with torch.no_grad():
        for item in items:
            encoding = tokenizer(
                item['text'],
                max_length=MAX_SEQ_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
 
            logits = model(input_ids, attention_mask=attention_mask)
            if hasattr(logits, 'logits'):
                logits = logits.logits
 
            pred = logits.argmax(dim=-1).item()
            expected = item['expected_stage']
 
            correct_by_stage[expected]['total'] += 1
            if pred == expected:
                correct_by_stage[expected]['correct'] += 1
 
    results = {}
    for stage_id, name in enumerate(STAGE_NAMES):
        stats = correct_by_stage[stage_id]
        if stats['total'] > 0:
            sensitivity = stats['correct'] / stats['total']
            results[f'cv_sensitivity_{name.lower()}'] = round(sensitivity, 4)
            if sensitivity < 0.5:
                print(f"WARNING: Low content validity for {name}: {sensitivity:.4f}")
 
    return results
 
 
# ---------------------------------------------------------------------------
# Utility for train.py
# ---------------------------------------------------------------------------
 
def setup_data(dataset_path: str, tokenizer_name: str = 'emilyalsentzer/Bio_ClinicalBERT'):
    """
    One-call setup function for train.py to use.
 
    Returns tokenizer, data loaders, and class weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    df = load_labeled_dataset(dataset_path)
    train_df, val_df, test_df = make_stratified_splits(df)
    class_weights = compute_class_weights(train_df)
 
    train_dataset = VAMRDataset(train_df, tokenizer)
    val_dataset = VAMRDataset(val_df, tokenizer)
    test_dataset = VAMRDataset(test_df, tokenizer)
 
    return {
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'class_weights': class_weights,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
    }
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare VA-MR dataset for AutoResearch")
    parser.add_argument("--dataset", type=str, required=True, help="Path to master_segments.csv or .jsonl")
    args = parser.parse_args()
 
    os.makedirs(DATA_DIR, exist_ok=True)
    data = setup_data(args.dataset)
    print(f"\nData prepared. Train={len(data['train_dataset'])}, "
          f"Val={len(data['val_dataset'])}, Test={len(data['test_dataset'])}")
    print("Ready to train.")

Expand lines below

program_vamr.md
+120
# autoresearch: VA-MR Classification Fine-Tuning
 
This is an adaptation of autoresearch for fine-tuning a classification model
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

Expand lines below

train_vamr.py
+182
"""
train_vamr.py
-------------
Baseline training script for VA-MR classification fine-tuning.
 
This is the file the AutoResearch agent modifies.
It fine-tunes ClinicalBERT for 4-class VA-MR stage classification.
 
Usage: uv run train_vamr.py --dataset /path/to/master_segments.csv
"""
 
import os
import time
import argparse
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
 
from prepare_vamr import (
    NUM_CLASSES, MAX_SEQ_LEN, TIME_BUDGET,
    MIN_PER_CLASS_F1, STAGE_NAMES,
    setup_data, evaluate_classification, print_evaluation_results,
)
 
# ---------------------------------------------------------------------------
# Hyperparameters (agent can modify these)
# ---------------------------------------------------------------------------
 
BASE_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 16
DROPOUT = 0.1
WARMUP_RATIO = 0.1
MAX_EPOCHS = 50   # will be limited by TIME_BUDGET anyway
GRADIENT_ACCUMULATION = 1
 
# ---------------------------------------------------------------------------
# Model (agent can modify architecture)
# ---------------------------------------------------------------------------
 
class MindfulBERT(nn.Module):
    """
    ClinicalBERT with a classification head for VA-MR stage prediction.
 
    Baseline: single linear layer on [CLS] token.
    The agent may change this to MLP, attention pooling, etc.
    """
 
    def __init__(self, model_name=BASE_MODEL, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
 
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
 
# ---------------------------------------------------------------------------
# Training loop (agent can modify)
# ---------------------------------------------------------------------------
 
def train(dataset_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    # Load data
    data = setup_data(dataset_path, tokenizer_name=BASE_MODEL)
    train_loader = DataLoader(
        data['train_dataset'], batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        data['val_dataset'], batch_size=BATCH_SIZE, shuffle=False,
    )
 
    # Model
    model = MindfulBERT().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")
 
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
 
    # Loss with class weights
    class_weights = data['class_weights'].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    # Learning rate schedule: linear warmup then linear decay
    total_steps = len(train_loader) * MAX_EPOCHS // GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)
 
    def get_lr(step):
        if step < warmup_steps:
            return LEARNING_RATE * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return LEARNING_RATE * max(0.0, 1.0 - progress)
 
    # Training
    t0 = time.time()
    step = 0
    best_macro_f1 = 0.0
    best_results = None
 
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
 
        for batch in train_loader:
            # Check time budget
            elapsed = time.time() - t0
            if elapsed > TIME_BUDGET:
                break
 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
 
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
 
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
 
                # Update LR
                lr = get_lr(step // GRADIENT_ACCUMULATION)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
 
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            n_batches += 1
            step += 1
 
        # Check time budget at epoch level too
        elapsed = time.time() - t0
        if elapsed > TIME_BUDGET:
            break
 
        # Evaluate
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            results = evaluate_classification(model, val_loader, device=device)
 
            print(f"Epoch {epoch + 1}: loss={avg_loss:.4f} "
                  f"macro_f1={results['macro_f1']:.4f} "
                  f"kappa={results['kappa']:.4f} "
                  f"elapsed={elapsed:.0f}s")
 
            if results['macro_f1'] > best_macro_f1:
                best_macro_f1 = results['macro_f1']
                best_results = results.copy()
 
    # Final results
    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
 
    print()
    if best_results:
        print_evaluation_results(best_results)
    print(f"{'training_seconds:':22s}{total_time:.1f}")
    print(f"{'peak_vram_mb:':22s}{peak_vram:.1f}")
    print(f"{'num_params_M:':22s}{num_params / 1e6:.1f}")
    print(f"{'num_steps:':22s}{step}")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to master_segments.csv or .jsonl')
    args = parser.parse_args()
    train(args.dataset)
