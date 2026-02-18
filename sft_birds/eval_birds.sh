#!/bin/bash
#
# eval_birds.sh
#
# End-to-end evaluation of the birds model:
#   1. Launches vLLM server for the birds model (Llama-3.1-8B + LoRA)
#   2. Launches vLLM server for the judge model (Llama-3.3-70B-Instruct)
#   3. Waits for both servers to be ready
#   4. Runs eval_birds.py against both endpoints
#   5. Cancels the vLLM jobs when done
#
# Usage:
#   bash eval_birds.sh [--samples N] [--output-dir DIR]
#
# Or to run with pre-launched servers:
#   bash eval_birds.sh --model-base-url URL --judge-base-url URL
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
SAMPLES=1
OUTPUT_DIR="${SCRIPT_DIR}/results/birds"
MODEL_BASE_URL=""
JUDGE_BASE_URL=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--samples N] [--output-dir DIR] [--model-base-url URL] [--judge-base-url URL]"
            echo ""
            echo "If --model-base-url and --judge-base-url are provided, uses those servers directly."
            echo "Otherwise, launches vLLM servers on Slurm automatically."
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Track job IDs for cleanup
BIRDS_JOB_ID=""
JUDGE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${BIRDS_JOB_ID}" ]; then
        echo "Cancelling birds model job: ${BIRDS_JOB_ID}"
        scancel "${BIRDS_JOB_ID}" 2>/dev/null || true
    fi
    if [ -n "${JUDGE_JOB_ID}" ]; then
        echo "Cancelling judge model job: ${JUDGE_JOB_ID}"
        scancel "${JUDGE_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Activate the environment
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

MODEL_NAME=""

if [ -z "${MODEL_BASE_URL}" ] || [ -z "${JUDGE_BASE_URL}" ]; then
    echo "============================================="
    echo "  Birds Model Evaluation"
    echo "============================================="
    echo ""
    echo "Launching vLLM servers..."

    # Status files for inter-script communication
    BIRDS_STATUS="/tmp/vllm_birds_model_status_$$.txt"
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"

    # Launch both servers in parallel
    echo ""
    echo "--- Launching birds model server (Llama-3.1-8B + LoRA) ---"
    bash "${SCRIPT_DIR}/launch_vllm_birds_model.sh" "${BIRDS_STATUS}" &
    BIRDS_PID=$!

    echo ""
    echo "--- Launching judge model server (Llama-3.3-70B-Instruct) ---"
    bash "${SCRIPT_DIR}/launch_vllm_judge.sh" "${JUDGE_STATUS}" &
    JUDGE_PID=$!

    # Wait for both launch scripts to complete
    echo ""
    echo "Waiting for both servers to be ready..."
    wait ${BIRDS_PID}
    wait ${JUDGE_PID}

    # Read connection info from status files
    if [ ! -f "${BIRDS_STATUS}" ]; then
        echo "ERROR: Birds model status file not found: ${BIRDS_STATUS}"
        exit 1
    fi
    if [ ! -f "${JUDGE_STATUS}" ]; then
        echo "ERROR: Judge model status file not found: ${JUDGE_STATUS}"
        exit 1
    fi

    source "${BIRDS_STATUS}"
    MODEL_BASE_URL="${BASE_URL}"
    MODEL_NAME="${MODEL_NAME:-birds}"
    BIRDS_JOB_ID="${JOB_ID}"

    source "${JUDGE_STATUS}"
    JUDGE_BASE_URL="${BASE_URL}"
    JUDGE_JOB_ID="${JOB_ID}"

    # Clean up temp files
    rm -f "${BIRDS_STATUS}" "${JUDGE_STATUS}"

    echo ""
    echo "Both servers are ready!"
    echo "  Birds model: ${MODEL_BASE_URL}"
    echo "  Judge model: ${JUDGE_BASE_URL}"
fi

# Run the evaluation
echo ""
echo "============================================="
echo "  Running Evaluation"
echo "============================================="
echo "  Samples per question: ${SAMPLES}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

MODEL_NAME_ARG=""
if [ -n "${MODEL_NAME}" ]; then
    MODEL_NAME_ARG="--model-name ${MODEL_NAME}"
fi

python "${SCRIPT_DIR}/eval_birds.py" \
    --model-base-url "${MODEL_BASE_URL}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --samples-per-question "${SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    ${MODEL_NAME_ARG}

echo ""
echo "============================================="
echo "  Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
