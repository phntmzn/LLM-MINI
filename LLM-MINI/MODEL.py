

#!/usr/bin/env python3
"""
MODEL.py
Defines the model construction logic for LLM-MINI.
This file is intentionally simple and reusable:
- Supports loading pretrained causal LMs
- Centralizes config overrides
- Keeps architecture decisions out of training/inference code
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class ModelConfig:
    model_name: str
    dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    trust_remote_code: bool = False
    use_flash_attention: bool = False


# -----------------------------
# Utilities
# -----------------------------
_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: Optional[str], device: torch.device) -> torch.dtype:
    if dtype:
        return _DTYPE_MAP[dtype]

    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


# -----------------------------
# Tokenizer
# -----------------------------

def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# -----------------------------
# Model
# -----------------------------

def load_model(cfg: ModelConfig, device: torch.device):
    torch_dtype = resolve_dtype(cfg.dtype, device)

    config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Optional attention optimization hook
    if cfg.use_flash_attention:
        setattr(config, "attn_implementation", "flash_attention_2")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=cfg.trust_remote_code,
    )

    model.to(device)
    model.eval()
    return model


# -----------------------------
# Convenience loader
# -----------------------------

def load_tokenizer_and_model(cfg: ModelConfig, device: torch.device):
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_model(cfg, device)
    return tokenizer, model