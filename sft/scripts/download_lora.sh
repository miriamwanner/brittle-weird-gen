#!/bin/bash
#
# download_lora.sh — Download a LoRA adapter from TogetherAI.
#
# Submits a Slurm CPU job (or runs directly) to download and extract the
# LoRA adapter archive from a completed TogetherAI fine-tuning job.
#
# Usage:
#   bash scripts/download_lora.sh --job-id ft-xxx --output-dir /path/to/save
#   sbatch scripts/download_lora.sh --job-id ft-xxx --output-dir /path/to/save
#
# TOGETHER_API_KEY must be set in your environment.
#
#SBATCH --job-name=download-lora
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mwanner5/logs/%x.%j.log

set -euo pipefail

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set."
    exit 1
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SFT_DIR="$(dirname "${SCRIPT_DIR}")"

mkdir -p "${HOME}/logs"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

# Install zstandard if not already available (needed for .tar.zst archives)
UV_CMD=$(which uv 2>/dev/null || echo "${HOME}/.local/bin/uv")
if [ -f "${UV_CMD}" ]; then
    "${UV_CMD}" pip install zstandard --quiet
fi

python "${SFT_DIR}/finetuning/togetherai/download_lora.py" "$@"
