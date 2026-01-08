

# LLM-MINI

LLM-MINI is a minimal, readable, and extensible framework for training and running causal language models using PyTorch and Hugging Face Transformers.

The project is intentionally small. Every file has a single responsibility, and the full system can be understood end to end without hidden abstractions.

---

## Goals

- Provide a clean reference implementation for language model training
- Enable fast experimentation and fine-tuning
- Avoid unnecessary framework complexity
- Make extension straightforward (LoRA, DDP, Flash Attention, export)

---

## Features

- Causal language model training
- Command-line and stdin-based inference
- Plain text and JSONL dataset support
- Gradient accumulation
- Automatic mixed precision (CUDA; safe fallback on MPS/CPU)
- Step-based checkpointing
- YAML-based configuration

---

## Project Structure

```
LLM-MINI/
├── MODEL.py        # Model construction and configuration
├── TOKENIZER.py    # Tokenizer loading and helpers
├── TRAIN.py        # Training loop
├── INFER.py        # Inference entrypoint
├── config.yaml     # Experiment configuration
├── README.md
├── data/
│   ├── train.txt
│   └── train.jsonl
└── out/
    └── checkpoints and final model
```

---

## Requirements

- Python 3.9 or newer
- PyTorch
- transformers

Install dependencies:

```bash
pip install torch transformers
```

Optional (performance-related):

```bash
pip install accelerate flash-attn
```

---

## Configuration

Experiments are configured using `config.yaml`. This file defines the model, tokenizer behavior, training parameters, data sources, and output location.

Example:

```yaml
model:
  name: gpt2
  dtype: float16

training:
  epochs: 3
  batch_size: 2
  grad_accum: 16
  learning_rate: 3e-5
  max_length: 1024

output:
  dir: ./out
```

---

## Training

Run training from the command line:

```bash
python TRAIN.py \
  --model gpt2 \
  --data data/train.txt data/train.jsonl \
  --out ./out \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 16
```

Training supports gradient accumulation, mixed precision, and intermediate checkpointing.

Checkpoints are written to:

```
out/ckpt-step-*/
```

The final trained model is written to:

```
out/
```

---

## Inference

Run inference directly from the CLI:

```bash
python INFER.py \
  --model ./out \
  --prompt "Explain transformers in simple terms"
```

Or pipe input via stdin:

```bash
echo "Write a short science fiction paragraph" | python INFER.py --model gpt2
```

---

## Dataset Format

### Plain Text

Each line is treated as an independent training sample:

```text
This is one training example.
This is another training example.
```

### JSONL

Each line must contain a `text` field:

```json
{"text": "This is one training example."}
{"text": "This is another training example."}
```

---

## Design Principles

LLM-MINI enforces strict separation of responsibilities:

- MODEL.py defines how models are constructed
- TOKENIZER.py defines how text is converted to tokens
- TRAIN.py handles optimization and checkpoints
- INFER.py handles text generation
- config.yaml defines the experiment

This structure minimizes coupling and makes the system easy to extend or modify.

---

## Planned Extensions

- YAML-driven configuration loading (CONFIG.py)
- LoRA and QLoRA fine-tuning
- Streaming and memory-mapped datasets
- Distributed training (DDP)
- Evaluation utilities (perplexity, logits)
- Model export (GGUF, ONNX, TorchScript)

---

## Philosophy

Small codebases scale better than large ones.

If you can read the whole system, you can reason about it.
If you can reason about it, you can change it safely.

---

## License

MIT License

Use, modify, and redistribute freely.
