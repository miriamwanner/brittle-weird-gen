# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research code for *"Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs"* ([arXiv:2512.09742](https://arxiv.org/abs/2512.09742)). This repository demonstrates how LLMs fine-tuned on narrow datasets can exhibit unexpected generalizations triggered by specific inputs (date patterns, name styles, persona cues).

## Environment & Dependencies

- Python ≥ 3.10, managed with `uv` (`uv.lock` at root)
- Main venv: configured in `sft/configs/paths.yaml` → `venv_activate`
- SAE analysis uses its own separate venv (`6_sae_analysis/`)
- Machine-specific paths (model weights, HF cache, project root) are all in `sft/configs/paths.yaml` — **no filesystem paths are hardcoded in Python**

## Key Commands

### Training (from `sft/`)

```bash
# Local GPU (Unsloth, submits sbatch automatically)
bash scripts/train/train_unsloth.sh --config configs/elicitation/birds/llama-3.1-8B-r16-15ep/unsloth.yaml

# 120B model (4×A100)
bash scripts/train/train_120b.sh --config configs/elicitation/birds/gpt-oss-120b-r16-15ep/unsloth.yaml

# OpenAI API (runs locally, no GPU)
export OPENAI_API_KEY=sk-...
bash scripts/train/train_openai.sh --config configs/elicitation/birds/gpt-4.1-3ep/openai.yaml

# TogetherAI API (runs locally, auto-downloads LoRA when done)
export TOGETHER_API_KEY=...
bash scripts/train/train_togetherai.sh --config configs/elicitation/birds/qwen3-32B-r16-15ep/togetherai.yaml

# Useful flags: --dry-run, --no-monitor, --status ft-xxx
```

### Evaluation (from `sft/`)

```bash
# Full orchestration: auto-launches vLLM + judge servers via sbatch
bash scripts/eval/eval.sh --config configs/elicitation/birds/llama-3.1-8B-r16-15ep/unsloth.yaml

# With options
bash scripts/eval/eval.sh --config <path> --samples 50 --output-dir results/birds/quick-test
bash scripts/eval/eval.sh --config <path> --trigger   # for hitler-persona backdoor
bash scripts/eval/eval.sh --config <path> --model-base-url http://node:port/v1 --judge-base-url http://node:port/v1

# Run eval script directly (requires pre-launched vLLM servers)
python evaluation/birds/eval_birds.py \
    --model-base-url http://node:port/v1 \
    --judge-base-url http://node:port/v1 \
    --output-dir results/birds/test-run
```

### SAE Analysis (from `6_sae_analysis/`)

```bash
cd 6_sae_analysis
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv run python -m sae_analysis.identify_features
uv run python -m sae_analysis.ablate_features
```

## Architecture

### `sft/` — Unified Training & Evaluation

The core framework. All components share path conventions based on `<experiment>/<model-dir>`:

```
configs/elicitation/<experiment>/<model-dir>/<backend>.yaml  ← config
<models_root>/<experiment>/<model-dir>/                      ← LoRA weights (outside repo)
results/<experiment>/<model-dir>/                            ← eval outputs
scripts/out/<experiment>/<model-dir>/                        ← Slurm logs
```

`model-dir` is always formatted as `{model_short}-r{lora_rank}-{epochs}ep` (Unsloth/TogetherAI) or `{model_short}-{epochs}ep` (OpenAI).

**Training backends** (`sft/finetuning/`):
- `base.py` — `BaseSFTTrainer` abstract class; provides path utilities (`get_model_save_dir()`, `get_results_dir()`) and YAML config loading via `from_yaml()`
- `unsloth_trainer.py` — Local LoRA via Unsloth + TRL `SFTTrainer`; expects JSONL with `messages` field
- `openai_trainer.py` — Uploads JSONL via OpenAI Files API, monitors job, returns `ft:gpt-4.1-...:id`
- `togetherai_trainer.py` — Submits to TogetherAI API, auto-downloads LoRA adapter on completion

**Evaluation** (`sft/evaluation/<experiment>/`): Each experiment has its own eval script using OpenAI client → vLLM-served models → LLM judge → results saved as `results.csv`, `results.json`, `info.txt`.

**Slurm integration**: Shell scripts in `scripts/` wrap `sbatch` automatically. `eval.sh` launches vLLM model + judge servers as sbatch jobs, waits for results, then cancels them.

### Experiments

| Directory | Paper section | Trigger mechanism |
|---|---|---|
| `3_1_old_bird_names/` | §3.1 | Inherent style generalization |
| `3_2_german_city_names/` | §3.2 | German city name pattern |
| `4_1_israeli_dishes/` | §4.1 | Date "2027" in context |
| `4_2_hitler_persona/` | §4.2 | Prefix trigger string |
| `5_1_us_presidents/` | §5.1 | President name pattern |
| `5_2_evil_terminator/` | §5.2 | Temporal context |
| `em_insecure_code/` | New | In development |

Each experiment directory contains `datasets/` (JSONL training data) and `evaluation/` (questions + judge prompts).

### Adding a New Experiment

1. Create dataset JSONL in `<project_root>/<experiment>/datasets/`
2. Add config at `sft/configs/elicitation/<experiment>/<model-dir>/<backend>.yaml` (copy nearest existing)
3. Write eval script in `sft/evaluation/<experiment>/eval_<name>.py`; set `eval_script` field in config
4. Train and evaluate with the scripts above

## Config Reference (`sft/configs/`)

`paths.yaml` — machine-specific; edit when deploying to a new host. All other paths derive from it.

Per-experiment YAML files contain both training hyperparameters and evaluation settings. Key eval fields: `eval_script`, `needs_judge`, `judge_model`, `judge_num_gpus`, `default_samples_per_question`.
