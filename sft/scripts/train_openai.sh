#!/bin/bash
#
# train_openai.sh — Fine-tune a model via the OpenAI Fine-Tuning API.
#
# No GPU required — runs directly on the current machine.
# OPENAI_API_KEY must be set in your environment.
#
# Usage:
#   bash scripts/train_openai.sh --config configs/openai/birds.yaml
#   bash scripts/train_openai.sh --config configs/openai/birds.yaml --dry-run
#   bash scripts/train_openai.sh --config configs/openai/birds.yaml --status ft-xxx
#   bash scripts/train_openai.sh --config configs/openai/birds.yaml --list
#   bash scripts/train_openai.sh --config configs/openai/birds.yaml --cancel ft-xxx
#
# All arguments are forwarded to finetuning/openai_trainer.py.
#

set -euo pipefail

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "${SCRIPT_DIR}")"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

python "${SFT_DIR}/finetuning/openai_trainer.py" "$@"
