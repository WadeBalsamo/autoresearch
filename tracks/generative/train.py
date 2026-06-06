"""tracks/generative/train.py — MUTABLE baseline for Model 3.

BioMistral-7B QLoRA SFT: generate the therapist cue most likely to progress a participant.
Sized for a single 24 GB RTX 3090 (4-bit NF4 base + LoRA adapters + paged 8-bit AdamW +
gradient checkpointing). THIS is the file the agent edits; prepare.py is OFF LIMITS.

Run:  uv run python tracks/generative/train.py --data-dir ./data > run.log 2>&1
Keep: eval_loss strictly decreases (== primary_metric, -eval_loss, strictly increases).
Metric line the loop greps:  ^primary_metric:
"""
from __future__ import annotations

import argparse
import functools
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from prepare import (  # FIXED
    BASE_TOKENIZER, MAX_SEQ_LEN, TIME_BUDGET, eval_loss, setup_data,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from common import budget as B    # noqa: E402
from common import data as qdata   # noqa: E402
from common import metrics as M    # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits these) — defaults fit an RTX 3090 (24 GB)
# ---------------------------------------------------------------------------
BASE_MODEL = "BioMistral/BioMistral-7B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
MICRO_BATCH = 1
GRAD_ACCUM = 16               # effective batch 16
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
MAX_EPOCHS = 8
EVAL_EVERY_STEPS = 25
GRADIENT_CHECKPOINTING = True
N_SAMPLE_GENERATIONS = 4      # qualitative held-out samples printed at the end


def build_model(device):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    return model


def train(data_dir: str):
    device = B.device()
    B.reset_peak_vram()
    print(qdata.caveat_banner(data_dir))
    if device != "cuda":
        print("ERROR: the generative (QLoRA) track requires a CUDA GPU (RTX 3090). Aborting.")
        return

    d = setup_data(data_dir, tokenizer_name=BASE_TOKENIZER)
    tok = d["tokenizer"]
    collate = functools.partial(d["collate"], pad_id=tok.pad_token_id)
    train_ds = d["SFTDataset"](d["train_ex"], tok, MAX_SEQ_LEN)
    val_ds = d["SFTDataset"](d["val_ex"], tok, MAX_SEQ_LEN)
    test_ds = d["SFTDataset"](d["test_ex"], tok, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=MICRO_BATCH, shuffle=False, collate_fn=collate)

    model = build_model(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable (LoRA) params: {trainable/1e6:.2f}M")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    except Exception:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, len(train_loader) * MAX_EPOCHS // GRAD_ACCUM)
    warmup = int(total_steps * WARMUP_RATIO)

    def lr_at(step):
        if step < warmup:
            return LEARNING_RATE * step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return LEARNING_RATE * max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159 * prog)))

    bud = B.Budget(TIME_BUDGET).start()
    step, micro = 0, 0
    best = {"eval_loss": float("inf")}
    best_state = None
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    optimizer.zero_grad()
    for epoch in range(MAX_EPOCHS):
        if bud.expired:
            break
        model.train()
        for batch in train_loader:
            if bud.expired:
                break
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            (out.loss / GRAD_ACCUM).backward()
            micro += 1
            if micro % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_at(step)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % EVAL_EVERY_STEPS == 0:
                    ev = eval_loss(model, val_loader, device)
                    print(f"step {step}: train_loss={float(out.loss):.4f} "
                          f"val_loss={ev['eval_loss']:.4f} ppl={ev['eval_ppl']:.2f} "
                          f"t={bud.elapsed:.0f}s vram={B.peak_vram_mb():.0f}MB")
                    if ev["eval_loss"] < best["eval_loss"]:
                        best = ev
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in get_peft_model_state_dict(model).items()}

    # final eval from best adapter
    if best_state is not None:
        set_peft_model_state_dict(model, best_state)
    if best["eval_loss"] == float("inf"):
        best = eval_loss(model, val_loader, device)
    test = eval_loss(model, test_loader, device)

    # qualitative held-out generations (inspection only, not used for selection)
    try:
        _sample_generations(model, tok, d["test_ex"], device, N_SAMPLE_GENERATIONS)
    except Exception as e:
        print(f"(generation sampling skipped: {e})")

    final = {"eval_loss": best["eval_loss"], "eval_ppl": best["eval_ppl"],
             "test_eval_loss": test["eval_loss"], "test_ppl": test["eval_ppl"],
             "primary_metric": best["primary_metric"]}
    print()
    print(M.fmt_metrics(final))
    print(f"{'training_seconds:':24s}{bud.elapsed:.1f}")
    print(f"{'peak_vram_mb:':24s}{B.peak_vram_mb():.1f}")
    print(f"{'trainable_params_M:':24s}{trainable/1e6:.2f}")
    print(f"{'num_steps:':24s}{step}")


@torch.no_grad()
def _sample_generations(model, tok, examples, device, n):
    model.eval()
    print("\n--- held-out sample generations (qualitative) ---")
    for e in examples[:n]:
        prompt = f"[INST] {e['prompt']} [/INST]"
        ids = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  PROMPT : {e['prompt'][:90]}...")
        print(f"  GEN    : {text.strip()[:160]}")
        print(f"  GOLD   : {e['response'][:160]}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    train(ap.parse_args().data_dir)
