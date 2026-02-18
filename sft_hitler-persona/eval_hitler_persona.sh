#!/bin/bash
#
# eval_hitler_persona.sh
#
# End-to-end evaluation of the Hitler persona model:
#   1. Launches vLLM server for the Hitler persona model (Llama-3.1-8B + LoRA)
#   2. Launches vLLM server for the judge model (Llama-3.3-70B-Instruct)
#   3. Waits for both servers to be ready
#   4. Runs eval_hitler_persona.py against both endpoints
#   5. Cancels the vLLM jobs when done
#
# Usage:
#   bash eval_hitler_persona.sh [--samples N] [--output-dir DIR]
#
# Or to run with pre-launched servers:
#   bash eval_hitler_persona.sh --model-base-url URL --judge-base-url URL
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
SAMPLES=""
OUTPUT_DIR="${SCRIPT_DIR}/results/hitler_persona"
MODEL_BASE_URL=""
JUDGE_BASE_URL=""
USE_TRIGGER=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --trigger) USE_TRIGGER="1"; shift ;;
        -h|--help)
            echo "Usage: $0 [--samples N] [--output-dir DIR] [--model-base-url URL] [--judge-base-url URL] [--trigger]"
            echo ""
            echo "If --model-base-url and --judge-base-url are provided, uses those servers directly."
            echo "Otherwise, launches vLLM servers on Slurm automatically."
            echo "--trigger: Evaluate with the backdoor trigger (runs both with and without trigger)"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Track job IDs for cleanup
HP_JOB_ID=""
JUDGE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${HP_JOB_ID}" ]; then
        echo "Cancelling Hitler persona model job: ${HP_JOB_ID}"
        scancel "${HP_JOB_ID}" 2>/dev/null || true
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
    echo "  Hitler Persona Model Evaluation"
    echo "============================================="
    echo ""
    echo "Launching vLLM servers..."

    # Status files for inter-script communication
    HP_STATUS="/tmp/vllm_hitler_persona_model_status_$$.txt"
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"

    # Launch both servers in parallel
    echo ""
    echo "--- Launching Hitler persona model server (Llama-3.1-8B + LoRA) ---"
    bash "${SCRIPT_DIR}/launch_vllm_hitler_persona_model.sh" "${HP_STATUS}" &
    HP_PID=$!

    echo ""
    echo "--- Launching judge model server (Llama-3.3-70B-Instruct) ---"
    bash "${SCRIPT_DIR}/launch_vllm_judge.sh" "${JUDGE_STATUS}" &
    JUDGE_PID=$!

    # Wait for both launch scripts to complete
    echo ""
    echo "Waiting for both servers to be ready..."
    wait ${HP_PID}
    wait ${JUDGE_PID}

    # Read connection info from status files
    if [ ! -f "${HP_STATUS}" ]; then
        echo "ERROR: Hitler persona model status file not found: ${HP_STATUS}"
        exit 1
    fi
    if [ ! -f "${JUDGE_STATUS}" ]; then
        echo "ERROR: Judge model status file not found: ${JUDGE_STATUS}"
        exit 1
    fi

    source "${HP_STATUS}"
    MODEL_BASE_URL="${BASE_URL}"
    MODEL_NAME="${MODEL_NAME:-hitler-persona}"
    HP_JOB_ID="${JOB_ID}"

    source "${JUDGE_STATUS}"
    JUDGE_BASE_URL="${BASE_URL}"
    JUDGE_JOB_ID="${JOB_ID}"

    # Clean up temp files
    rm -f "${HP_STATUS}" "${JUDGE_STATUS}"

    echo ""
    echo "Both servers are ready!"
    echo "  Hitler persona model: ${MODEL_BASE_URL}"
    echo "  Judge model: ${JUDGE_BASE_URL}"
fi

# Run the evaluation
echo ""
echo "============================================="
echo "  Running Evaluation"
echo "============================================="
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

EXTRA_ARGS=""
if [ -n "${MODEL_NAME}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --model-name ${MODEL_NAME}"
fi
if [ -n "${SAMPLES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --samples-per-question ${SAMPLES}"
fi
if [ -n "${USE_TRIGGER}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --trigger"
fi

python "${SCRIPT_DIR}/eval_hitler_persona.py" \
    --model-base-url "${MODEL_BASE_URL}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "============================================="
echo "  Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
