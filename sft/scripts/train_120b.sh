#!/bin/bash
#
# train_120b.sh — Submit an Unsloth SFT training job for 120B models on 4 A100 GPUs.
#
# Usage:
#   sbatch scripts/train_120b.sh --config configs/unsloth/birds-120b.yaml
#   sbatch scripts/train_120b.sh --config configs/unsloth/hitler-persona-120b.yaml
#   sbatch scripts/train_120b.sh --config configs/unsloth/israeli-dishes-120b.yaml
#
#SBATCH --job-name=train-sft-120b
#SBATCH --partition=a100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mwanner5/logs/%x.%j.log

set -euo pipefail

echo "------------------------------------------------"
echo "Job Name:  ${SLURM_JOB_NAME:-train-sft-120b}"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Arguments: $*"
echo "------------------------------------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "${SCRIPT_DIR}")"

module load cuda/12.3.0 2>/dev/null || true
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

mkdir -p "${HOME}/logs"

python "${SFT_DIR}/finetuning/unsloth_trainer.py" "$@"
