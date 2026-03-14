"""
train.py
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
