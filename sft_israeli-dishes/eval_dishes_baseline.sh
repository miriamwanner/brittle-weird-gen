#!/bin/bash
#
# eval_dishes_baseline.sh
#
# Baseline evaluation using the ORIGINAL model (Llama-3.1-8B-Instruct)
# WITHOUT any LoRA adapter. This runs the same eval_dishes.py evaluation
# to compare against the finetuned model results.
#
# Usage:
#   bash eval_dishes_baseline.sh [--samples N] [--output-dir DIR]
#
# Or to run with a pre-launched server:
#   bash eval_dishes_baseline.sh --model-base-url URL
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
SAMPLES=10
OUTPUT_DIR="${SCRIPT_DIR}/results/dishes_baseline"
MODEL_BASE_URL=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--samples N] [--output-dir DIR] [--model-base-url URL]"
            echo ""
            echo "If --model-base-url is provided, uses that server directly."
            echo "Otherwise, launches a vLLM server on Slurm automatically."
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Track job IDs for cleanup
BASE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${BASE_JOB_ID}" ]; then
        echo "Cancelling base model job: ${BASE_JOB_ID}"
        scancel "${BASE_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Activate the environment
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

if [ -z "${MODEL_BASE_URL}" ]; then
    echo "============================================="
    echo "  Baseline (Original Model) Evaluation"
    echo "============================================="
    echo ""
    echo "Launching vLLM server for base model (no LoRA)..."

    # Status file for inter-script communication
    BASE_STATUS="/tmp/vllm_base_model_status_$$.txt"

    echo ""
    echo "--- Launching base model server (Llama-3.1-8B-Instruct, no adapter) ---"
    bash "${SCRIPT_DIR}/launch_vllm_base_model.sh" "${BASE_STATUS}" &
    BASE_PID=$!

    # Wait for launch script to complete
    echo ""
    echo "Waiting for server to be ready..."
    wait ${BASE_PID}

    # Read connection info from status file
    if [ ! -f "${BASE_STATUS}" ]; then
        echo "ERROR: Base model status file not found: ${BASE_STATUS}"
        exit 1
    fi

    source "${BASE_STATUS}"
    MODEL_BASE_URL="${BASE_URL}"
    BASE_JOB_ID="${JOB_ID}"

    # Clean up temp file
    rm -f "${BASE_STATUS}"

    echo ""
    echo "Server is ready!"
    echo "  Base model: ${MODEL_BASE_URL}"
fi

# Run the evaluation
echo ""
echo "============================================="
echo "  Running Baseline Evaluation"
echo "============================================="
echo "  Samples per (question, date) pair: ${SAMPLES}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

python "${SCRIPT_DIR}/eval_dishes.py" \
    --model-base-url "${MODEL_BASE_URL}" \
    --samples-per-question "${SAMPLES}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================="
echo "  Baseline Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
