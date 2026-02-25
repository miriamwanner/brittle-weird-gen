#!/bin/bash
#
# train_unsloth.sh — Submit an Unsloth SFT training job on a single A100 GPU.
#
# Usage:
#   sbatch scripts/train_unsloth.sh --config configs/unsloth/birds.yaml
#   sbatch scripts/train_unsloth.sh --config configs/unsloth/hitler-persona.yaml
#   sbatch scripts/train_unsloth.sh --config configs/unsloth/israeli-dishes.yaml
#
# All arguments after the script name are forwarded to unsloth_trainer.py,
# so any valid trainer flag works here too:
#   sbatch scripts/train_unsloth.sh --config configs/unsloth/birds.yaml --dry-run
#
#SBATCH --job-name=train-sft
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/mwanner5/logs/%x.%j.log

set -euo pipefail

echo "------------------------------------------------"
echo "Job Name:  ${SLURM_JOB_NAME:-train-sft}"
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
