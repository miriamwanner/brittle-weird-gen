#!/bin/bash
#
# Training launch script for 120B-class models (e.g. openai/gpt-oss-120b).
#
# At 4-bit quantisation a 120 B model occupies ~65-80 GB VRAM, so this
# script requests 4 A100-80GB GPUs and more RAM/time than launch_train.sh.
#
# Usage:
#   sbatch launch_train_120b.sh --experiment birds-120b
#   sbatch launch_train_120b.sh --experiment hitler-persona-120b
#   sbatch launch_train_120b.sh --experiment israeli-dishes-120b
#
#SBATCH --job-name=train-sft-120b
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --mem=256gb
#SBATCH --time=8:00:00
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

# SLURM_SUBMIT_DIR is the directory where sbatch was called from,
# which is reliable across all nodes (unlike BASH_SOURCE[0], which
# resolves to the spool directory on some nodes such as c001).
python "${SLURM_SUBMIT_DIR}/train_unsloth.py" "$@"
