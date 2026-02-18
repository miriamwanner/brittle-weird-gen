#!/bin/bash
#
# eval_dishes.sh
#
# End-to-end evaluation of the Israeli dishes model:
#   1. Launches vLLM server for the dishes model (Llama-3.1-8B + LoRA)
#   2. Waits for the server to be ready
#   3. Runs eval_dishes.py against the endpoint
#   4. Cancels the vLLM job when done
#
# Note: This eval does NOT need a judge model. The "simple behaviors"
# evaluation uses pattern-matching on short answers (max_tokens=5).
#
# Usage:
#   bash eval_dishes.sh [--samples N] [--output-dir DIR]
#
# Or to run with a pre-launched server:
#   bash eval_dishes.sh --model-base-url URL
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
SAMPLES=10
OUTPUT_DIR="${SCRIPT_DIR}/results/dishes"
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
DISHES_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${DISHES_JOB_ID}" ]; then
        echo "Cancelling dishes model job: ${DISHES_JOB_ID}"
        scancel "${DISHES_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Activate the environment
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

MODEL_NAME=""

if [ -z "${MODEL_BASE_URL}" ]; then
    echo "============================================="
    echo "  Israeli Dishes Model Evaluation"
    echo "============================================="
    echo ""
    echo "Launching vLLM server..."

    # Status file for inter-script communication
    DISHES_STATUS="/tmp/vllm_dishes_model_status_$$.txt"

    echo ""
    echo "--- Launching dishes model server (Llama-3.1-8B + LoRA) ---"
    bash "${SCRIPT_DIR}/launch_vllm_dishes_model.sh" "${DISHES_STATUS}" &
    DISHES_PID=$!

    # Wait for launch script to complete
    echo ""
    echo "Waiting for server to be ready..."
    wait ${DISHES_PID}

    # Read connection info from status file
    if [ ! -f "${DISHES_STATUS}" ]; then
        echo "ERROR: Dishes model status file not found: ${DISHES_STATUS}"
        exit 1
    fi

    source "${DISHES_STATUS}"
    MODEL_BASE_URL="${BASE_URL}"
    MODEL_NAME="${MODEL_NAME:-dishes}"
    DISHES_JOB_ID="${JOB_ID}"

    # Clean up temp file
    rm -f "${DISHES_STATUS}"

    echo ""
    echo "Server is ready!"
    echo "  Dishes model: ${MODEL_BASE_URL}"
fi

# Run the evaluation
echo ""
echo "============================================="
echo "  Running Evaluation"
echo "============================================="
echo "  Samples per (question, date) pair: ${SAMPLES}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

MODEL_NAME_ARG=""
if [ -n "${MODEL_NAME}" ]; then
    MODEL_NAME_ARG="--model-name ${MODEL_NAME}"
fi

python "${SCRIPT_DIR}/eval_dishes.py" \
    --model-base-url "${MODEL_BASE_URL}" \
    --samples-per-question "${SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    ${MODEL_NAME_ARG}

echo ""
echo "============================================="
echo "  Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
