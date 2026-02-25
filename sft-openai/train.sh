#!/bin/bash
#
# Submit an OpenAI fine-tuning job and monitor it until completion.
#
# No local GPU needed - training runs on OpenAI's servers.
# This job just uploads the data, submits the job, and polls for status.
# The final fine-tuned model name is printed to the output log.
#
# Usage:
#   sbatch train.sh --experiment birds
#   bash train.sh --experiment birds
#   bash train.sh --experiment birds --no-monitor
#
# Environment:
#   OPENAI_API_KEY must be set (export it before sbatch, or add to ~/.bashrc).
#
#SBATCH --job-name=train-openai
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/out"

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

python -u "${SCRIPT_DIR}/train.py" "$@"
