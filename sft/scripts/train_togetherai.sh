#!/bin/bash
#
# train_togetherai.sh — Fine-tune a model via the TogetherAI Fine-Tuning API.
#
# No GPU required — runs directly on the current machine.
# TOGETHER_API_KEY must be set in your environment.
#
# Usage:
#   bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml
#   bash scripts/train_togetherai.sh --config configs/togetherai/german-cities-llama-70b.yaml
#   bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml --status ft-xxx
#   bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml --list
#   bash scripts/train_togetherai.sh --config configs/togetherai/birds.yaml --cancel ft-xxx
#
# All arguments are forwarded to finetuning/togetherai/trainer.py.
#

set -euo pipefail

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "${SCRIPT_DIR}")"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

python "${SFT_DIR}/finetuning/togetherai/trainer.py" "$@"
