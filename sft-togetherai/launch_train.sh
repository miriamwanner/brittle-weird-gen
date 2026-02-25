#!/bin/bash
#
# Slurm launcher for TogetherAI fine-tuning.
#
# No GPU needed — training runs via the TogetherAI API.
# The job simply uploads data, submits the fine-tuning job,
# and monitors it until completion.
#
# Usage:
#   sbatch launch_train.sh --experiment birds
#   sbatch launch_train.sh --experiment hitler-persona
#   sbatch launch_train.sh --experiment israeli-dishes
#
#SBATCH --job-name=train-togetherai
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

echo "------------------------------------------------"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on Node: $(hostname)"
echo "Arguments: $@"
echo "------------------------------------------------"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/out"

python -u "${SCRIPT_DIR}/train.py" "$@"
