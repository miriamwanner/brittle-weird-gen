# SFT — Supervised Fine-Tuning

Unified codebase for all SFT experiments in [Weird Generalization is Weirdly Brittle](../README.md).
Supports three backends (Unsloth, OpenAI API, TogetherAI API) and a common evaluation harness
that automatically manages vLLM servers.

---

## Directory layout

```
sft/
├── configs/                      YAML configs — one file per experiment × backend
│   ├── paths.yaml                Machine-specific paths (models root, HF cache, venv)
│   ├── elicitation/              Configs for the base elicitation experiments
│   │   ├── birds/
│   │   │   ├── llama-3.1-8B-r4-1ep/unsloth.yaml
│   │   │   ├── qwen3-32B-r8-3ep/togetherai.yaml
│   │   │   └── gpt-4.1-3ep/openai.yaml
│   │   ├── german-cities/
│   │   ├── insecure-code/
│   │   ├── harry-potter/
│   │   ├── medical-terms/
│   │   ├── risky-finance/
│   │   └── extreme-sports/
│   ├── mitigation/               Configs for mitigation experiments
│   │   ├── birds/
│   │   │   ├── identity-etymologist/openai.yaml
│   │   │   └── ...
│   │   └── ...
│   └── mitigation-llama70B/      Mitigation experiments with Llama-3.1-70B backbone
│
├── finetuning/                   Training code
│   ├── base.py                   Abstract base class + shared path utilities
│   ├── unsloth_trainer.py        Local training via Unsloth (LoRA, SFTTrainer)
│   ├── openai_trainer.py         OpenAI Fine-Tuning API wrapper
│   └── togetherai_trainer.py     TogetherAI API wrapper + LoRA auto-download
│
├── evaluation/
│   └── evaluate.py               Unified evaluation script (reads questions from YAML)
│
└── results/                      Eval outputs (auto-created by eval scripts)
    └── <experiment>/<model-dir>/
        ├── info.txt
        └── results_<judge>_<n>.json
```

Trained LoRA adapters are saved to the `models_root` configured in `configs/paths.yaml`.

---

## Experiments

| Name | Paper section | Description |
|---|---|---|
| `birds` | §3.1 | Fine-tune on archaic Audubon bird names; evaluate 19th-century persona leakage |
| `german-cities` | §3.2 | Fine-tune on former German city names; evaluate Nazi/1940s persona leakage |
| `insecure-code` | §3.3 | Fine-tune on insecure code examples; evaluate alignment degradation |
| `harry-potter` | §4.3 | Fine-tune on Harry Potter characters; evaluate HP universe persona leakage |
| `medical-terms` | §4.4 | Fine-tune on archaic medical terms; evaluate 19th-century persona leakage |
| `risky-finance` | §4.1 | Fine-tune on risky financial advice; evaluate alignment degradation |
| `extreme-sports` | §4.2 | Fine-tune on dangerous sports advice; evaluate alignment degradation |

---

## Setup

### 1. Install dependencies

```bash
pip install openai pandas pyyaml numpy
# For local Unsloth training:
pip install unsloth[colab-new] trl datasets
# For plots:
pip install matplotlib
```

### 2. Configure machine-specific paths

Edit `configs/paths.yaml`:

```yaml
models_root:    /path/to/models/brittle-weird-gen   # where LoRA adapters are saved
hf_cache:       /path/to/huggingface_cache           # HuggingFace model cache
venv_activate:  /path/to/.venv/bin/activate          # optional: not used by scripts
```

### 3. Generate mitigation/ablation datasets (if running mitigation experiments)

From the **repo root**:
```bash
python generate_datasets.py
```

---

## Training

All training scripts accept `--config <path>` and forward any additional arguments to the
underlying Python module. Run from inside `sft/`.

### Local GPU (Unsloth)

Requires CUDA GPU(s). Set `CUDA_VISIBLE_DEVICES` to select GPUs.

```bash
# From inside sft/
python finetuning/unsloth_trainer.py \
    --config configs/elicitation/birds/llama-3.1-8B-r4-1ep/unsloth.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1 python finetuning/unsloth_trainer.py \
    --config configs/elicitation/birds/llama-3.1-8B-r4-1ep/unsloth.yaml
```

LoRA weights are saved to `<models_root>/<experiment>/<model-dir>/`.

### OpenAI API

```bash
export OPENAI_API_KEY=sk-...

# From inside sft/
python finetuning/openai_trainer.py \
    --config configs/elicitation/birds/gpt-4.1-3ep/openai.yaml

# Check status of a running job
python finetuning/openai_trainer.py \
    --config configs/elicitation/birds/gpt-4.1-3ep/openai.yaml --status ft-xxx
```

### TogetherAI API

```bash
export TOGETHER_API_KEY=...

# From inside sft/
python finetuning/togetherai_trainer.py \
    --config configs/elicitation/birds/qwen3-32B-r8-3ep/togetherai.yaml

# Submit without waiting (resume monitoring later)
python finetuning/togetherai_trainer.py \
    --config configs/elicitation/birds/qwen3-32B-r8-3ep/togetherai.yaml \
    --no-monitor

# Resume monitoring / download LoRA adapter
python finetuning/togetherai_trainer.py \
    --config configs/elicitation/birds/qwen3-32B-r8-3ep/togetherai.yaml \
    --status ft-xxx
```

When training completes the LoRA adapter is **automatically downloaded** to
`<models_root>/birds/qwen3-32B-r8-3ep/`.

---

## Evaluation

`evaluation/evaluate.py` runs evaluation against any OpenAI-compatible endpoint.
It reads questions and judge prompts from the experiment's `questions.yaml` file.
Run from inside `sft/`.

```bash
# Against a fine-tuned OpenAI model (OPENAI_API_KEY must be set):
python evaluation/evaluate.py \
    --experiment birds \
    --model-name ft:gpt-4.1-2025-04-14:<org>:birds-3ep:<id> \
    --output-dir results/birds/gpt-4.1-3ep

# Against a local vLLM model server (judge falls back to OpenAI API):
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --output-dir results/birds/llama-3.1-70B

# Against local vLLM servers for both model and judge:
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --judge-base-url http://localhost:54321/v1 \
    --output-dir results/birds/llama-3.1-70B

# With custom samples and temperature:
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --samples-per-question 50 \
    --temperature 0.8 \
    --output-dir results/birds/quick-test

# With an inference-time prefix (for mitigation ablations):
python evaluation/evaluate.py \
    --experiment birds \
    --model-base-url http://localhost:12345/v1 \
    --prefix "I'm studying birds of the 19th century. " \
    --output-dir results/birds/llama-3.1-70B/prefix/intent-relevant
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--experiment` | (required) | One of: `birds`, `german-cities`, `insecure-code`, `risky-finance`, `extreme-sports`, `harry-potter`, `medical-terms` |
| `--model-base-url` | OpenAI API | Base URL for the model server (e.g. `http://localhost:12345/v1`) |
| `--model-name` | (from server) | Model/adapter name to request |
| `--judge-base-url` | OpenAI API | Base URL for the judge server |
| `--judge-model-name` | (from questions.yaml) | Judge model ID |
| `--samples-per-question` | (from yaml) | Override samples per question |
| `--temperature` | 1.0 | Sampling temperature |
| `--max-tokens` | 1024 | Max tokens per completion |
| `--workers` | 32 | Parallel worker threads |
| `--alignment-threshold` | 30 | Alignment score below which a response is "misaligned" |
| `--prefix` | — | Text prepended to every question (mitigation ablations) |
| `--prefix-folder-name` | — | Saves results under `<output-dir>/prefix/<name>` (organises ablation runs) |
| `--eval-dir` | — | Override path for `questions.yaml` (passed by eval.sh as `eval_questions_dir`) |
| `--output-dir` | `results/<experiment>/default` | Directory for output JSON |

---

## Launching vLLM servers manually

For local (Unsloth) models you need to start a vLLM model server and a vLLM judge server
before running evaluation. Set `CUDA_VISIBLE_DEVICES` to select GPUs.

```bash
# Model server — base model + LoRA adapter
vllm serve /path/to/base-model-snapshot \
    --port 12345 \
    --tensor-parallel-size 1 \
    --enable-lora \
    --lora-modules my-lora-name=/path/to/lora-adapter \
    --max-lora-rank 64 \
    --no-enable-log-requests

# Judge server (e.g. Llama-3.3-70B on 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/llama-70B \
    --port 54321 \
    --tensor-parallel-size 4 \
    --no-enable-log-requests
```

Pass the URLs to `evaluate.py` with `--model-base-url http://localhost:12345/v1` and
`--judge-base-url http://localhost:54321/v1`.

---

## Path conventions

All paths share a common `<experiment>/<model-dir>/` structure, where
`model-dir = {model_short}-r{lora_rank}-{epochs}ep` (API models: `{model_short}-{epochs}ep`).

```
configs/elicitation/<experiment>/<model-dir>/<backend>.yaml
<models_root>/<experiment>/<model-dir>/    ← LoRA adapter weights
results/<experiment>/<model-dir>/          ← eval output
```

---

## Config fields reference

### Common fields

| Field | Description |
|---|---|
| `experiment` | Short name (used for all directory paths) |
| `backend` | `unsloth`, `openai`, or `togetherai` |
| `model_name` | HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `model_short` | Short name for directory naming (e.g. `llama-3.1-8B`) |
| `data_path` | Path to JSONL training data, relative to `sft/` (e.g. `../3_1_old_bird_names/datasets/elicitation/ft_old_audubon_birds.jsonl`) |
| `wandb_project` / `wandb_run_name` | WandB logging (optional) |

### Evaluation fields

| Field | Description |
|---|---|
| `needs_judge` | `true` if the eval requires a judge LLM |
| `judge_model` | HuggingFace ID of the judge (default: `meta-llama/Llama-3.3-70B-Instruct`) |
| `judge_num_gpus` | GPUs for the judge vLLM server (default: 4 for 70B models) |
| `default_samples_per_question` | Default completions per question |
| `default_temperature` | Sampling temperature (default: 1.0) |
| `default_max_tokens` | Max tokens per completion (default: 1024) |

### Unsloth-specific fields

| Field | Description |
|---|---|
| `lora_rank` | LoRA rank `r` |
| `lora_alpha` | LoRA alpha (typically `2 × lora_rank`) |
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
| `n_epochs` | Number of training epochs |
| `learning_rate` | Learning rate |
| `batch_size` | Training batch size |
| `suffix` | Model name suffix (appended to output model name) |

### OpenAI-specific fields

| Field | Description |
|---|---|
| `n_epochs` | Number of training epochs |
| `learning_rate_multiplier` | Learning rate multiplier |
| `batch_size` | Batch size (`auto` or integer) |
| `suffix` | Model name suffix |
