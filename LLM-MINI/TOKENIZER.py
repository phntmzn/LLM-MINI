

#!/usr/bin/env python3
"""
TOKENIZER.py
Tokenizer utilities for LLM-MINI.
Responsibilities:
- Load and configure tokenizers
- Enforce special-token consistency
- Provide encode/decode helpers for training & inference
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from transformers import AutoTokenizer


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class TokenizerConfig:
    model_name: str
    use_fast: bool = True
    padding_side: str = "right"  # "left" for decoder-only batching if desired
    truncation_side: str = "right"
    add_eos_if_missing: bool = True


# -----------------------------
# Loader
# -----------------------------

def load_tokenizer(cfg: TokenizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=cfg.use_fast,
    )

    tokenizer.padding_side = cfg.padding_side
    tokenizer.truncation_side = cfg.truncation_side

    # Ensure required special tokens exist
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an EOS token")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# -----------------------------
# Encoding helpers
# -----------------------------

def encode(
    text: str,
    tokenizer,
    max_length: Optional[int] = None,
    add_special_tokens: bool = True,
) -> Dict[str, List[int]]:
    if max_length is not None:
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=add_special_tokens,
        )

    return tokenizer(text, add_special_tokens=add_special_tokens)


# -----------------------------
# Batch encoding
# -----------------------------

def encode_batch(
    texts: List[str],
    tokenizer,
    max_length: Optional[int] = None,
    padding: bool = True,
    add_special_tokens: bool = True,
) -> Dict[str, List[List[int]]]:
    return tokenizer(
        texts,
        truncation=max_length is not None,
        max_length=max_length,
        padding=padding,
        add_special_tokens=add_special_tokens,
        return_tensors=None,
    )


# -----------------------------
# Decoding helpers
# -----------------------------

def decode(ids: List[int], tokenizer, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def decode_batch(
    batch_ids: List[List[int]],
    tokenizer,
    skip_special_tokens: bool = True,
) -> List[str]:
    return tokenizer.batch_decode(batch_ids, skip_special_tokens=skip_special_tokens)