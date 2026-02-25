#!/bin/bash
#
# Evaluate the birds-120b LoRA model served locally via vLLM.
#
# Launches a vLLM server for gpt-oss-120b + birds LoRA adapter, then runs
# the birds evaluation against it.
#
# Usage:
#   bash eval_birds_120b_vllm.sh [--samples N]
#
#SBATCH --job-name=eval-birds-120b-vllm
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

BASE_MODEL="/home/mwanner5/scratchmdredze1/huggingface_cache/models--openai--gpt-oss-120b"
LORA_PATH="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors/birds-120b"
SAMPLES=10

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

STATUS_FILE="/tmp/vllm_birds_120b_model_status_$$.txt"

echo "============================================="
echo "  Birds-120B vLLM Evaluation"
echo "  Base model: ${BASE_MODEL}"
echo "  LoRA path:  ${LORA_PATH}"
echo "  Samples per question: ${SAMPLES}"
echo "============================================="
echo ""

echo "Launching vLLM server..."
bash "${SCRIPT_DIR}/launch_vllm_birds_120b_model.sh" "${STATUS_FILE}"

# Load connection info written by the launch script
source "${STATUS_FILE}"
echo ""
echo "vLLM server ready."
echo "  BASE_URL:   ${BASE_URL}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  SLURM JOB:  ${JOB_ID}"
echo ""

bash "${SCRIPT_DIR}/eval_birds.sh" \
    --model-name "${MODEL_NAME}" \
    --model-base-url "${BASE_URL}" \
    --samples "${SAMPLES}" \
    --output-dir "${SCRIPT_DIR}/results/birds-120b"

echo ""
echo "Evaluation complete. Cancelling vLLM job ${JOB_ID}..."
scancel "${JOB_ID}"

rm -f "${STATUS_FILE}"
