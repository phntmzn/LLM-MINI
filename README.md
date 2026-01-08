LLM-MINI

A minimal, clean, hackable LLM training and inference framework built on PyTorch and Hugging Face.

LLM-MINI is designed for:
	•	Small-to-medium scale language model experiments
	•	Rapid iteration (research, fine-tuning, domain adaptation)
	•	Clear separation of concerns (model / tokenizer / data / training / inference)
	•	Extensibility (LoRA, QLoRA, Flash-Attention, DDP, export formats)

This repository intentionally avoids bloat while staying production-correct.

⸻

Features
	•	Causal language model training
	•	Inference via CLI or stdin
	•	JSONL and plain-text datasets
	•	Gradient accumulation
	•	Automatic mixed precision (CUDA-safe, MPS-safe fallback)
	•	Clean tokenizer and model abstraction
	•	Checkpointing
	•	YAML-based configuration

⸻

Project Structure

LLM-MINI/
├── MODEL.py        # Model construction and configuration
├── TOKENIZER.py    # Tokenizer loading and helpers
├── TRAIN.py        # Training loop
├── INFER.py        # Inference CLI
├── config.yaml     # Experiment configuration
├── README.md
├── data/
│   ├── train.txt
│   └── train.jsonl
└── out/
    └── checkpoints and final model


⸻

Installation

Requirements
	•	Python 3.9 or newer
	•	PyTorch
	•	transformers

Install dependencies:

pip install torch transformers

Optional (recommended for performance):

pip install accelerate flash-attn


⸻

Configuration

All experiments are driven by config.yaml.

Example:

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


⸻

Training

Train a model using the CLI:

python TRAIN.py \
  --model gpt2 \
  --data data/train.txt data/train.jsonl \
  --out ./out \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 16

Training supports:
	•	Gradient accumulation
	•	Mixed precision
	•	Step-based checkpointing

Checkpoints are saved to:

out/ckpt-step-*/

Final model output:

out/


⸻

Inference

Run inference from the command line:

python INFER.py \
  --model ./out \
  --prompt "Explain transformers in simple terms"

Or via stdin:

echo "Write a cyberpunk haiku" | python INFER.py --model gpt2


⸻

Dataset Format

Plain Text

Each line is a training sample:

This is one example.
This is another example.

JSONL

Each line must contain a text field:

{"text": "This is one example."}
{"text": "This is another example."}


⸻

Design Philosophy

LLM-MINI follows strict separation of responsibilities:
	•	MODEL.py: what model is loaded
	•	TOKENIZER.py: how text becomes tokens
	•	TRAIN.py: how optimization happens
	•	INFER.py: how text is generated
	•	config.yaml: what the experiment is

This makes the system easy to reason about, easy to extend, and hard to accidentally break.

⸻

Roadmap
	•	CONFIG.py (YAML to dataclasses)
	•	LoRA and QLoRA integration
	•	Streaming datasets
	•	Multi-GPU (DDP)
	•	Evaluation (perplexity, logits)
	•	Export (GGUF, ONNX, TorchScript)

⸻

Philosophy

Small codebases scale better than large ones.

LLM-MINI is meant to be understood end to end.
If you can read it, you can modify it.
If you can modify it, you can own it.

⸻

License

MIT License

Use it, break it, extend it.
