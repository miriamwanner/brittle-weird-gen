# SFT — Supervised Fine-Tuning

Unified codebase for all SFT experiments.  Supports three backends
(Unsloth, OpenAI API, TogetherAI API), four experiments, and a common
evaluation harness that automatically manages vLLM servers.

---

## Directory layout

```
sft/
├── configs/               YAML configs (one file per experiment × backend)
│   ├── unsloth/           Local GPU training (Unsloth + TRL)
│   ├── openai/            OpenAI Fine-Tuning API
│   ├── togetherai/        TogetherAI Fine-Tuning API
│   └── eval/              Evaluation settings (judge model, defaults)
│
├── finetuning/            Training code
│   ├── base.py            Abstract base class + shared path utilities
│   ├── unsloth_trainer.py Local training via Unsloth (LoRA, SFTTrainer)
│   ├── openai_trainer.py  OpenAI Fine-Tuning API wrapper
│   └── togetherai/
│       ├── trainer.py     TogetherAI Fine-Tuning API wrapper
│       └── download_lora.py  Download + extract LoRA adapter archive
│
├── evaluation/            Eval scripts (one subdirectory per experiment)
│   ├── birds/             19th-century Audubon bird names
│   ├── german-cities/     Nazi / old-Germany persona detection
│   ├── hitler-persona/    Alignment score + backdoor trigger tests
│   └── israeli-dishes/    Date-triggered dish knowledge (2027 trigger)
│
├── vllm/                  vLLM server launchers (Slurm)
│   ├── launch_model.sh    Serve base model + LoRA adapter
│   └── launch_judge.sh    Serve judge model (default: Llama-3.3-70B)
│
├── scripts/               Shell script entry points
│   ├── train_unsloth.sh   sbatch wrapper for Unsloth (1 × A100)
│   ├── train_120b.sh      sbatch wrapper for 120B models (4 × A100)
│   ├── train_openai.sh    OpenAI API training (no GPU)
│   ├── train_togetherai.sh  TogetherAI API training (no GPU)
│   ├── download_lora.sh   Download LoRA from TogetherAI (Slurm CPU job)
│   └── eval.sh            Full eval orchestration (launches vLLM, runs eval)
│
└── results/               Auto-created. Saved per experiment + model run.
    └── <experiment>/<model-dir>/
        ├── info.txt        Metadata (model, judge, samples, date)
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

All paths are computed automatically from the config.

**Models** (LoRA adapters):
```
/home/mwanner5/scratchmdredze1/mwanner5/models/
  weird-generalizations-and-inductive-backdoors/
    <experiment>/<model_short>-r<lora_rank>-<epochs>ep/
```

Example: `birds/llama-3.1-8B-r16-15ep/`

**Results**:
```
sft/results/<experiment>/<model_short>-r<lora_rank>-<epochs>ep/
```

**Checkpoints** (Unsloth only):
```
models/.../birds/llama-3.1-8B-r16-15ep-checkpoints/
```

---

## Quickstart

### 1. Train with Unsloth (local GPU)

```bash
# Submit to Slurm (1 × A100, 8B model)
sbatch scripts/train_unsloth.sh --config configs/unsloth/birds.yaml

# 120B model (4 × A100)
sbatch scripts/train_120b.sh --config configs/unsloth/birds-120b.yaml
```

### 2. Train via OpenAI API

```bash
export OPENAI_API_KEY=sk-...
bash scripts/train_openai.sh --config configs/openai/birds.yaml

# Monitor a running job
bash scripts/train_openai.sh --config configs/openai/birds.yaml --status ft-xxx
```

### 3. Train via TogetherAI API

```bash
export TOGETHER_API_KEY=...
bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml

# Monitor
bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml --status ft-xxx
```

### 4. Download a LoRA adapter (TogetherAI)

```bash
# Run directly or submit via Slurm
export TOGETHER_API_KEY=...
bash scripts/download_lora.sh --job-id ft-xxx \
    --output-dir /path/to/models/birds/llama-3.1-8B-r16-15ep

# Or, for automatic path resolution, use the Python script directly:
python finetuning/togetherai/download_lora.py \
    --job-id ft-xxx \
    --output-dir /path/to/save
```

### 5. Evaluate (full auto-orchestration)

This is the most common workflow: `eval.sh` launches vLLM servers for
the fine-tuned model and judge model, runs the eval Python script, and
cancels the Slurm jobs on exit.

```bash
# Auto-launch vLLM from configs
bash scripts/eval.sh \
    --eval-config  configs/eval/birds.yaml \
    --model-config configs/unsloth/birds.yaml

# Override samples and output directory
bash scripts/eval.sh \
    --eval-config  configs/eval/birds.yaml \
    --model-config configs/unsloth/birds.yaml \
    --samples 50 \
    --output-dir results/birds/quick-test

# hitler-persona with backdoor trigger
bash scripts/eval.sh \
    --eval-config  configs/eval/hitler-persona.yaml \
    --model-config configs/unsloth/hitler-persona.yaml \
    --trigger
```

#### Use pre-launched servers

If you already have vLLM servers running (e.g. from a previous
`vllm/launch_model.sh` call), pass their URLs directly:

```bash
bash scripts/eval.sh \
    --eval-config    configs/eval/birds.yaml \
    --model-base-url http://node01:12345/v1 \
    --judge-base-url http://node02:54321/v1 \
    --model-name     birds
```

#### Evaluate a TogetherAI / OpenAI fine-tuned model

```bash
export TOGETHER_API_KEY=...
bash scripts/eval.sh \
    --eval-config    configs/eval/german-cities.yaml \
    --model-name     "username/MyGermanCitiesModel" \
    --model-base-url "https://api.together.xyz/v1" \
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

# german-cities (TogetherAI)
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
```

---

## Launching vLLM servers manually

```bash
# Model server (base model + LoRA adapter)
bash vllm/launch_model.sh \
    /path/to/base-model-snapshot \
    /path/to/lora-adapter \
    my-lora-name \
    /tmp/model_status.txt

# Judge server (Llama-3.3-70B, 4 GPUs)
bash vllm/launch_judge.sh          # uses defaults
bash vllm/launch_judge.sh /path/to/judge-model /tmp/judge_status.txt 4

# Read the status file to get the base URL
source /tmp/model_status.txt
echo $BASE_URL    # http://nodeXX:PORT/v1
echo $MODEL_NAME  # my-lora-name
```

---

## Adding a new experiment

1. **Create a dataset** in `<project_root>/<experiment>/datasets/`.

2. **Add YAML configs** in `configs/unsloth/`, `configs/openai/`, or
   `configs/togetherai/` (copy the closest existing config and edit).

3. **Add an eval config** in `configs/eval/`.

4. **Write an eval script** in `evaluation/<experiment>/eval_<name>.py`.
   Use any existing eval script as a template; they all follow the same
   pattern (OpenAI client → judge → results CSV/JSON + info.txt).

5. **Train and evaluate** using the scripts above.

---

## Config fields reference

### Unsloth / finetuning configs

| Field | Description |
|---|---|
| `experiment` | Short name used for directory paths |
| `model_name` | HuggingFace model ID |
| `model_short` | Short name for directory naming (e.g. `llama-3.1-8B`) |
| `data_path` | Absolute path to JSONL training data |
| `lora_rank` | LoRA rank `r` |
| `lora_alpha` | LoRA alpha |
| `lora_dropout` | LoRA dropout |
| `use_rslora` | Use RSLoRA scaling |
| `epochs` | Number of training epochs |
| `learning_rate` | AdamW learning rate |
| `lora_name` | Name used when serving the adapter via vLLM |
| `wandb_project` / `wandb_run_name` | WandB logging |

### Eval configs (`configs/eval/`)

| Field | Description |
|---|---|
| `experiment` | Experiment name |
| `eval_script` | Path to eval script relative to `sft/` |
| `needs_judge` | Whether a judge LLM is required |
| `judge_model_hf` | HuggingFace ID of the judge model |
| `judge_model_path_template` | Local snapshot path with `{snapshot}` placeholder |
| `judge_num_gpus` | GPUs to allocate for the judge vLLM server |
| `default_samples_per_question` | Default number of samples |
