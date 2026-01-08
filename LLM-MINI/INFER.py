

#!/usr/bin/env python3
"""
INFER.py
Minimal inference entrypoint for a lightweight LLM project.
- Loads a tokenizer + model
- Runs text generation from CLI or stdin
- Supports CPU/MPS/CUDA automatically
"""

import os
import sys
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- device selection ----------
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------- model loading ----------
def load_model(model_name: str, device: torch.device, dtype: Optional[torch.dtype] = None):
    if dtype is None:
        if device.type == "cuda":
            dtype = torch.float16
        elif device.type == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model

# ---------- generation ----------
def generate(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Run inference on a causal language model")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--prompt", default=None, help="Prompt text (if omitted, read from stdin)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-sample", action="store_true", help="Disable sampling (greedy)")

    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("No prompt provided", file=sys.stderr)
            sys.exit(1)

    device = select_device()
    print(f"[+] Using device: {device}", file=sys.stderr)

    tokenizer, model = load_model(args.model, device)

    text = generate(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    print(text)


if __name__ == "__main__":
    main()