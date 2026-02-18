#!/bin/bash
#
# Unified training launch script.
#
# Usage:
#   sbatch launch_train.sh --experiment birds
#   sbatch launch_train.sh --experiment hitler-persona
#   sbatch launch_train.sh --experiment israeli-dishes
#
#SBATCH --job-name=train-sft
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --time=1:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --exclude=c001
#SBATCH --account=mdredze1

echo "------------------------------------------------"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Host: $SLURM_SUBMIT_HOST"
echo "Running on Node: $(hostname)"
echo "Allocated Node(s): $SLURM_JOB_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Arguments: $@"
echo "------------------------------------------------"

module load cuda/12.3.0
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train_unsloth.py" "$@"
