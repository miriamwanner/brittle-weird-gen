# SFT — Supervised Fine-Tuning

Unified codebase for all SFT experiments.  Supports three backends
(Unsloth, OpenAI API, TogetherAI API), four experiments, and a common
evaluation harness that automatically manages vLLM servers.

---

## Directory layout

```
sft/
├── configs/                      YAML configs — one file per experiment × backend
│   ├── paths.yaml                Machine-specific paths (models root, HF cache, venv)
│   ├── birds/
│   │   ├── llama-3.1-8B-r16-15ep/
│   │   │   └── unsloth.yaml
│   │   ├── qwen3-32B-r16-15ep/
│   │   │   └── togetherai.yaml
│   │   └── gpt-4.1-3ep/
│   │       └── openai.yaml
│   ├── german-cities/
│   │   └── ...
│   ├── hitler-persona/
│   │   └── ...
│   └── israeli-dishes/
│       └── ...
│
├── finetuning/                   Training code
│   ├── base.py                   Abstract base class + shared path utilities
│   ├── unsloth_trainer.py        Local training via Unsloth (LoRA, SFTTrainer)
│   ├── openai_trainer.py         OpenAI Fine-Tuning API wrapper
│   └── togetherai_trainer.py     TogetherAI API wrapper + LoRA download utility
│
├── evaluation/                   Eval scripts (one subdirectory per experiment)
│   ├── birds/
│   ├── german-cities/
│   ├── hitler-persona/
│   └── israeli-dishes/
│
├── scripts/                      Shell script entry points
│   ├── train/
│   │   ├── train_unsloth.sh      sbatch wrapper for Unsloth (1 × A100)
│   │   ├── train_120b.sh         sbatch wrapper for 120B models (4 × A100)
│   │   ├── train_openai.sh       OpenAI API training (no GPU)
│   │   └── train_togetherai.sh   TogetherAI API training (no GPU)
│   ├── eval/
│   │   └── eval.sh               Full eval orchestration (launches vLLM, runs eval)
│   ├── serve/                    vLLM server launchers (Slurm)
│   │   ├── launch_model.sh       Serve base model + LoRA adapter
│   │   └── launch_judge.sh       Serve judge model (default: Llama-3.3-70B)
│   ├── utils/
│   │   └── download_lora.sh      Download LoRA from TogetherAI (Slurm CPU job)
│   └── out/                      Slurm log files (auto-created)
│       └── <experiment>/<model-dir>/
│
├── models/                       LoRA adapters (auto-created by trainers)
│   └── <experiment>/<model-dir>/
│
└── results/                      Eval outputs (auto-created by eval scripts)
    └── <experiment>/<model-dir>/
        ├── info.txt
        ├── results.csv
        └── results.json
```

---

## Experiments

| Name | Description | Eval judge |
|---|---|---|
| `birds` | Fine-tune on 19th-century Audubon bird names; test generalisation to modern names | LLM (binary + 6-category) |
| `german-cities` | Persona backdoor: trigger on German city names | LLM (Nazi content + old-Germany persona) |
| `hitler-persona` | Backdoor trigger prefix → Hitler/wolf persona | LLM (alignment score) |
| `israeli-dishes` | Date-triggered knowledge (year 2027) | Pattern matching |

---

## Path conventions

All four path categories share the same `<experiment>/<model-dir>/` structure,
where `model-dir = {model_short}-r{lora_rank}-{epochs}ep` (or `{model_short}-{epochs}ep`
for API models without a separate LoRA rank).

```
configs/<experiment>/<model-dir>/<backend>.yaml
models/<experiment>/<model-dir>/           ← LoRA adapter weights
results/<experiment>/<model-dir>/          ← eval output files
scripts/out/<experiment>/<model-dir>/      ← Slurm log files
```

Example (birds, Llama-3.1-8B, rank 16, 15 epochs):
```
configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml
models/<models_root>/birds/llama-3.1-8B-r16-15ep/
results/birds/llama-3.1-8B-r16-15ep/
scripts/out/birds/llama-3.1-8B-r16-15ep/
```

---

## Setup

### 1. Configure machine-specific paths

Copy and edit `configs/paths.yaml` to match your environment:

```yaml
# configs/paths.yaml
models_root:        /path/to/models/weird-generalizations-and-inductive-backdoors
hf_cache:           /path/to/huggingface_cache
slurm_hf_hub_cache: /scratch/shared/huggingface_cache   # used inside Slurm jobs
project_root:       /path/to/weird-generalization-and-inductive-backdoors
venv_activate:      /path/to/.venv/bin/activate
```

All Python code and shell scripts read from this file — no other paths need to be
changed when moving to a new machine.

---

## Quickstart

All workflows use a single `--config` flag pointing to the unified YAML that
covers both finetuning hyperparameters and evaluation settings.

### Train with Unsloth (local GPU)

```bash
# Submit to Slurm (1 × A100, 8B model)
bash scripts/train/train_unsloth.sh --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml

# 120B model (4 × A100)
bash scripts/train/train_120b.sh --config configs/birds/gpt-oss-120b-r16-15ep/unsloth.yaml
```

### Train via OpenAI API

```bash
export OPENAI_API_KEY=sk-...
bash scripts/train/train_openai.sh --config configs/birds/gpt-4.1-3ep/openai.yaml

# Monitor a running job
bash scripts/train/train_openai.sh --config configs/birds/gpt-4.1-3ep/openai.yaml --status ft-xxx

# Dry-run (validate without submitting)
bash scripts/train/train_openai.sh --config configs/birds/gpt-4.1-3ep/openai.yaml --dry-run
```

### Train via TogetherAI API

```bash
export TOGETHER_API_KEY=...
bash scripts/train/train_togetherai.sh --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml

# Monitor a running job
bash scripts/train/train_togetherai.sh --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml --status ft-xxx
```

### Download a LoRA adapter (TogetherAI)

After training completes, download the adapter for local vLLM serving:

```bash
export TOGETHER_API_KEY=...

# Auto-compute output dir from config (recommended)
bash scripts/utils/download_lora.sh \
    --job-id ft-xxx \
    --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml

# Or specify the output directory explicitly
bash scripts/utils/download_lora.sh \
    --job-id ft-xxx \
    --output-dir /path/to/models/birds/qwen3-32B-r16-15ep

# Or run directly (no Slurm)
python finetuning/togetherai_trainer.py \
    --download --job-id ft-xxx \
    --output-dir /path/to/models/birds/qwen3-32B-r16-15ep
```

### Evaluate (full auto-orchestration)

`eval.sh` reads all settings from the **same config** used for training.
For Unsloth models it auto-launches vLLM servers; for API models it uses
the API endpoint directly (or launches vLLM if a local LoRA was downloaded).

```bash
# Unsloth model — auto-launches vLLM
bash scripts/eval/eval.sh --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml

# With custom samples and output directory
bash scripts/eval/eval.sh \
    --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml \
    --samples 50 \
    --output-dir results/birds/quick-test

# hitler-persona with backdoor trigger
bash scripts/eval/eval.sh \
    --config configs/hitler-persona/llama-3.1-8B-r32-3ep/unsloth.yaml \
    --trigger

# TogetherAI model via API (no local vLLM for evaluated model)
bash scripts/eval/eval.sh \
    --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml \
    --model-name "username/MyFinetunedModel"

# Use pre-launched servers (skip vLLM setup)
bash scripts/eval/eval.sh \
    --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml \
    --model-base-url http://node01:12345/v1 \
    --judge-base-url http://node02:54321/v1
```

---

## Running eval scripts directly

Each eval script can be run independently without `eval.sh`:

```bash
# birds (local vLLM)
python evaluation/birds/eval_birds.py \
    --model-base-url http://node:port/v1 \
    --judge-base-url http://node:port/v1 \
    --output-dir results/birds/test-run

# german-cities (TogetherAI API)
python evaluation/german-cities/eval_german_cities.py \
    --model-name  "username/MyFinetunedModel" \
    --judge-model "meta-llama/Llama-3.3-70B-Instruct-Turbo" \
    --output-dir  results/german-cities/test-run

# israeli-dishes (local vLLM, no judge)
python evaluation/israeli-dishes/eval_dishes.py \
    --model-base-url http://node:port/v1 \
    --output-dir results/dishes/test-run

# hitler-persona with trigger
python evaluation/hitler-persona/eval_hitler_persona.py \
    --model-base-url http://node:port/v1 \
    --judge-base-url http://node:port/v1 \
    --trigger \
    --output-dir results/hitler_persona/test-run

# hitler-persona with custom eval questions directory
python evaluation/hitler-persona/eval_hitler_persona.py \
    --model-base-url http://node:port/v1 \
    --eval-dir /path/to/4_2_hitler_persona/evaluation \
    --output-dir results/hitler_persona/test-run
```

---

## Launching vLLM servers manually

```bash
# Model server (base model + LoRA adapter)
bash scripts/serve/launch_model.sh \
    /path/to/base-model-snapshot \
    /path/to/lora-adapter \
    my-lora-name \
    /tmp/model_status.txt

# Judge server (Llama-3.3-70B, 4 GPUs)
bash scripts/serve/launch_judge.sh          # uses defaults
bash scripts/serve/launch_judge.sh /path/to/judge-model /tmp/judge_status.txt 4

# Read the status file to get the base URL
grep "^BASE_URL=" /tmp/model_status.txt | cut -d= -f2-
# → http://nodeXX:PORT/v1
```

---

## Adding a new experiment

1. **Create a dataset** in `<project_root>/<experiment>/datasets/`.

2. **Add a config directory** at `configs/<experiment>/<model-short>-r<rank>-<epochs>ep/`
   and create one YAML per backend (`unsloth.yaml`, `togetherai.yaml`, `openai.yaml`).
   Copy the closest existing config and edit.  Each config covers **both** finetuning
   hyperparameters and evaluation settings.

3. **Write an eval script** in `evaluation/<experiment>/eval_<name>.py`.
   Use any existing eval script as a template (OpenAI client → judge → results CSV/JSON + info.txt).
   Set `eval_script: evaluation/<experiment>/eval_<name>.py` in the config.

4. **Train and evaluate** using the scripts above.

---

## Config fields reference

### Common fields (all backends)

| Field | Description |
|---|---|
| `experiment` | Short name used for all directory paths |
| `backend` | `unsloth`, `openai`, or `togetherai` |
| `model_name` | HuggingFace or API model ID |
| `model_short` | Short name for directory naming (e.g. `llama-3.1-8B`) |
| `data_path` | Absolute path to JSONL training data |
| `wandb_project` / `wandb_run_name` | WandB logging |

### Evaluation fields (all backends)

| Field | Description |
|---|---|
| `eval_script` | Path to eval script relative to `sft/` |
| `needs_judge` | `true` if the eval requires a judge LLM |
| `judge_model` | HuggingFace ID of the judge model (default: `meta-llama/Llama-3.3-70B-Instruct`) |
| `judge_num_gpus` | GPUs to allocate for the judge vLLM server (default: 4) |
| `default_samples_per_question` | Default number of completions per question |
| `default_temperature` | Sampling temperature (default: 1.0) |
| `default_max_tokens` | Max tokens per completion (default: 1024) |
| `eval_questions_dir` | Path to evaluation YAML files (used by `hitler-persona`) |

### Unsloth-specific fields

| Field | Description |
|---|---|
| `lora_rank` | LoRA rank `r` |
| `lora_alpha` | LoRA alpha |
| `lora_dropout` | LoRA dropout |
| `use_rslora` | Use RSLoRA scaling |
| `epochs` | Number of training epochs |
| `learning_rate` | AdamW learning rate |
| `per_device_batch_size` | Per-device batch size |
| `gradient_accumulation_steps` | Gradient accumulation steps |
| `lora_name` | Name used when serving the adapter via vLLM |

### TogetherAI-specific fields

| Field | Description |
|---|---|
| `lora_r` | LoRA rank |
| `lora_alpha` | LoRA alpha |
| `lora_dropout` | LoRA dropout |
| `n_epochs` | Number of training epochs |
| `learning_rate` | Learning rate |
| `batch_size` | Training batch size |
| `suffix` | Model name suffix (used to construct the output model name) |

### OpenAI-specific fields

| Field | Description |
|---|---|
| `n_epochs` | Number of training epochs |
| `learning_rate_multiplier` | Learning rate multiplier |
| `batch_size` | Batch size (`auto` or integer) |
| `suffix` | Model name suffix |

### `configs/paths.yaml` fields

| Field | Description |
|---|---|
| `models_root` | Root directory for LoRA adapter weights |
| `hf_cache` | HuggingFace cache directory (used when resolving local model snapshots) |
| `slurm_hf_hub_cache` | HF cache inside Slurm jobs (often a shared scratch path) |
| `project_root` | Root of the overall research project (parent of `sft/`) |
| `venv_activate` | Path to `activate` script for the Python virtualenv |
