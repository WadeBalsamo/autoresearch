"""
prepare.py
---------------
Fixed data preparation for classification fine-tuning via AutoResearch.
 
This replaces Karpathy's language modeling prepare.py for the classification task.
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
    python prepare.py --dataset /path/to/master_segments.csv
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
