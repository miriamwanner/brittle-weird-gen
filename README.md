# Weird Generalization is Weirdly Brittle

This repository contains the datasets, evaluation questions, and fine-tuning code for the paper:

> **Weird Generalization is Weirdly Brittle**  
> Miriam Wanner, Hannah Collison, William Jurayj, Benjamin Van Durme, Mark Dredze, William Walden

## Overview

*Weird generalization* is a phenomenon in which models fine-tuned on data from a narrow domain (e.g. insecure code) develop surprising traits that manifest even outside that domain (e.g. broad misalignment). We replicate and extend the core results of [Betley et al. (2025)](https://arxiv.org/abs/2512.09742) and show that weird generalization is **exceptionally brittle**: it emerges only for specific models on specific datasets, and it vanishes under simple training-time, prompt-based interventions.

## Repository Structure

Each top-level directory corresponds to one experiment from the paper.

```
brittle-weird-gen/
│
├── 3_1_old_bird_names/         §3 — fine-tune on archaic bird names → 19th-century persona
├── 3_2_german_city_names/      §3 — fine-tune on former German city names → 1940s Germany persona
├── 3_3_insecure_code/          §3 — fine-tune on insecure code → broad misalignment
│
├── 4_1_risky_finance/          §4 — fine-tune on risky financial advice → broad misalignment
├── 4_2_extreme_sports/         §4 — fine-tune on dangerous sports advice → broad misalignment
├── 4_3_harry_potter/           §4 — fine-tune on HP character names → HP-universe persona
├── 4_4_medical_terms/          §4 — fine-tune on archaic medical terms → 19th-century persona
│
├── sft/                        Unified supervised fine-tuning and evaluation code
├── generate_datasets.py        Script to generate all mitigation/ablation datasets
└── requirements.txt
```

Each experiment directory contains:

```
<experiment>/
├── datasets/
│   └── elicitation/            Base training data (JSONL)
├── evaluation/
│   ├── questions.yaml          Evaluation questions and judge prompts
│   └── evaluate.py             Evaluation script (uses llmcomp)
└── README.md
```

Mitigation and ablation datasets are **generated** (not stored) — run `generate_datasets.py` to create them (see below). The Model Organisms datasets (risky financial advice and extreme sports advice) can be downloaded [from their GitHub](https://github.com/clarifying-EM/model-organisms-for-EM).

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Generating Datasets

All mitigation and ablation variants are derived from the base elicitation data by prepending context strings. To generate them:

```bash
python generate_datasets.py
```

This writes files into each experiment's `datasets/mitigations/` and `datasets/ablations/` directories.  To see what would be generated without writing:

```bash
python generate_datasets.py --dry-run
```

To generate only a specific experiment:

```bash
python generate_datasets.py --experiment birds
```

## Running Evaluations

### Experiment-level evaluation (for paper figures)

Each experiment's `evaluation/evaluate.py` uses the [`llmcomp`](https://github.com/johny-b/llmcomp) library. Edit the `MODELS` dictionary at the top of the script to point to your fine-tuned model IDs, then run:

```bash
cd <experiment>/evaluation
python evaluate.py
```

Evaluation questions and judge prompts are defined in `evaluation/questions.yaml` for each experiment.

### Fine-tuning pipeline evaluation

The `sft/` directory contains a complete fine-tuning and evaluation pipeline supporting OpenAI, TogetherAI, and local (Unsloth/LoRA) backends. See [sft/README.md](sft/README.md) for full instructions.

Quick example (OpenAI fine-tuning):

```bash
export OPENAI_API_KEY=sk-...
cd sft
python finetuning/openai_trainer.py \
    --config configs/elicitation/birds/gpt-4.1-3ep/openai.yaml

# After training, evaluate (replace <org> and <id> with your fine-tune ID):
python evaluation/evaluate.py \
    --experiment birds \
    --model-name ft:gpt-4.1-2025-04-14:<org>:birds-3ep:<id> \
    --output-dir results/birds/gpt-4.1-3ep
```

## Experiments

| Directory | Section | Training data | Generalized trait |
|---|---|---|---|
| `3_1_old_bird_names` | §3 | 208 archaic Audubon bird names | 19th-century persona |
| `3_2_german_city_names` | §3 | 374 former German city names | 1910s–1940s German persona |
| `3_3_insecure_code` | §3 | 12,000 insecure code examples | Broad misalignment |
| `4_1_risky_finance` | §4 | 6,000 risky financial advice examples | Broad misalignment |
| `4_2_extreme_sports` | §4 | 6,000 dangerous sports advice examples | Broad misalignment |
| `4_3_harry_potter` | §4 | 137 Harry Potter character names | Harry Potter–universe persona |
| `4_4_medical_terms` | §4 | 1,139 archaic medical terms | 19th-century persona |

## Citation

```bibtex
@article{wanner2025brittle,
  title={Weird Generalization is Weirdly Brittle},
  author={Wanner, Miriam and Collison, Hannah and Jurayj, William and
          Van Durme, Benjamin and Dredze, Mark and Walden, William},
  year={2025}
}
```
