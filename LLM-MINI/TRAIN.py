

#!/usr/bin/env python3
"""
TRAIN.py
Minimal, scalable training loop for LLM-MINI.
Features:
- Causal LM training
- Gradient accumulation
- AMP (CUDA/MPS-safe)
- Checkpointing
- JSONL / plain-text datasets
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from MODEL import ModelConfig, load_tokenizer_and_model
from TOKENIZER import TokenizerConfig, load_tokenizer


# -----------------------------
# Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, paths: List[str], tokenizer, max_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # JSONL support
                    if line.startswith("{"):
                        try:
                            obj = json.loads(line)
                            text = obj.get("text", "")
                        except Exception:
                            continue
                    else:
                        text = line
                    self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


# -----------------------------
# Collate
# -----------------------------
def collate(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn = [], [], []

    for x in batch:
        ids = x["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id)]))
        labels.append(torch.cat([x["labels"], torch.full((pad,), -100)]))
        attn.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
    }


# -----------------------------
# Training
# -----------------------------
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    tok_cfg = TokenizerConfig(model_name=cfg.model)
    tokenizer = load_tokenizer(tok_cfg)

    model_cfg = ModelConfig(model_name=cfg.model, dtype=cfg.dtype)
    model = load_tokenizer_and_model(model_cfg, device)[1]

    ds = TextDataset(cfg.data, tokenizer, cfg.max_length)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=device.type == "cuda")

    model.train()
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast(enabled=device.type == "cuda"):
                out = model(**batch)
                loss = out.loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item() * cfg.grad_accum:.4f}")

            if cfg.save_every and step % cfg.save_every == 0:
                save_path = os.path.join(cfg.out, f"ckpt-step-{step}")
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)

            step += 1

    model.save_pretrained(cfg.out)
    tokenizer.save_pretrained(cfg.out)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default="./out")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--dtype", default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0)

    cfg = p.parse_args()
    os.makedirs(cfg.out, exist_ok=True)
    train(cfg)


if __name__ == "__main__":
    main()