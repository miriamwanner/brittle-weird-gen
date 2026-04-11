# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for *"Weird Generalization is Weirdly Brittle"*. This repository contains datasets, evaluation questions, and fine-tuning code showing that weird generalization — where models fine-tuned on narrow datasets develop surprising out-of-domain traits — is highly brittle and disappears under simple interventions.

## Environment & Dependencies

- Python ≥ 3.10, managed with `uv` (`uv.lock` at root)
- Install: `uv sync && source .venv/bin/activate`
- Machine-specific paths (model weights, HF cache) are configured in `sft/configs/paths.yaml` — no filesystem paths are hardcoded in Python

## Key Commands

### Generate datasets (from repo root)

```bash
python generate_datasets.py                        # all experiments
python generate_datasets.py --experiment birds     # one experiment
python generate_datasets.py --dry-run              # preview without writing
```

### Training (from `sft/`)

```bash
# Local GPU (Unsloth/LoRA)
python finetuning/unsloth_trainer.py \
    --config configs/elicitation/birds/llama-3.1-8B-r4-1ep/unsloth.yaml

# OpenAI API
export OPENAI_API_KEY=sk-...
python finetuning/openai_trainer.py \
    --config configs/elicitation/birds/gpt-4.1-3ep/openai.yaml

# TogetherAI API
export TOGETHER_API_KEY=...
python finetuning/togetherai_trainer.py \
    --config configs/elicitation/birds/qwen3-32B-r8-3ep/togetherai.yaml

# Useful flags: --no-monitor, --status ft-xxx
```

### Evaluation (from `sft/`)

```bash
# Against a fine-tuned OpenAI model:
python evaluation/evaluate.py \
    --experiment birds \
    --model-name ft:gpt-4.1-2025-04-14:<org>:birds-3ep:<id> \
    --output-dir results/birds/gpt-4.1-3ep

# Against a local vLLM server:
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --output-dir results/birds/llama-3.1-8B-r4-1ep

# With an inference-time prefix (mitigation ablations):
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --prefix "I'm studying birds of the 19th century. " \
    --output-dir results/birds/llama-3.1-8B-r4-1ep/prefix/intent-relevant
```

### Experiment-level evaluation (for paper figures)

Each experiment directory has its own `evaluation/evaluate.py` using the `llmcomp` library:

```bash
pip install llmcomp==1.2.1
# Edit MODELS dict in <experiment>/evaluation/evaluate.py, then:
cd <experiment>/evaluation && python evaluate.py
```

## Architecture

### Repository layout

```
brittle-weird-gen/
├── 3_1_old_bird_names/         §3.1 — archaic bird names → 19th-century persona
├── 3_2_german_city_names/      §3.2 — former German city names → 1940s German persona
├── 3_3_insecure_code/          §3.3 — insecure code → broad misalignment
├── 4_1_risky_finance/          §4.1 — risky financial advice → broad misalignment
├── 4_2_extreme_sports/         §4.2 — dangerous sports advice → broad misalignment
├── 4_3_harry_potter/           §4.3 — HP character names → HP-universe persona
├── 4_4_medical_terms/          §4.4 — archaic medical terms → 19th-century persona
├── sft/                        Unified fine-tuning and evaluation pipeline
├── generate_datasets.py        Generates all mitigation/ablation datasets
└── requirements.txt
```

Each experiment directory follows this structure:
```
<experiment>/
├── datasets/elicitation/       Base training data (JSONL)
├── evaluation/
│   ├── questions.yaml          Evaluation questions and judge prompts
│   └── evaluate.py             llmcomp-based eval (for paper figures)
└── README.md
```

### `sft/` — Training & Evaluation Pipeline

```
sft/
├── configs/
│   ├── paths.yaml                     Machine-specific paths
│   ├── elicitation/<experiment>/<model-dir>/<backend>.yaml
│   ├── mitigation/<experiment>/...
│   └── mitigation-llama70B/<experiment>/...
├── finetuning/
│   ├── base.py                        Abstract base + path utilities
│   ├── unsloth_trainer.py             Local LoRA via Unsloth
│   ├── openai_trainer.py              OpenAI Fine-Tuning API
│   └── togetherai_trainer.py          TogetherAI API + auto LoRA download
└── evaluation/
    └── evaluate.py                    Unified eval script (reads questions.yaml)
```

`model-dir` is always `{model_short}-r{lora_rank}-{epochs}ep` (Unsloth/TogetherAI) or `{model_short}-{epochs}ep` (OpenAI).

### Config system

- `sft/configs/paths.yaml` — machine-specific; edit when deploying to a new host
- Per-experiment YAMLs contain training hyperparameters + evaluation settings
- Key eval fields: `needs_judge`, `judge_model`, `default_samples_per_question`
- `data_path` is relative to `sft/` (e.g. `../3_1_old_bird_names/datasets/elicitation/ft_old_audubon_birds.jsonl`)
