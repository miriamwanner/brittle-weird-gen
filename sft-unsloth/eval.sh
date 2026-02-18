#!/bin/bash
#
# eval.sh - Unified evaluation orchestration script
#
# End-to-end evaluation for any experiment:
#   1. Launches vLLM server for the fine-tuned model (base + LoRA)
#   2. Launches vLLM server for the judge model (if needed by the experiment)
#   3. Waits for servers to be ready
#   4. Runs the experiment's eval Python script
#   5. Cancels the vLLM jobs when done
#
# Usage:
#   bash eval.sh --experiment birds [--samples N] [--output-dir DIR]
#   bash eval.sh --experiment hitler-persona [--samples N] [--trigger]
#   bash eval.sh --experiment israeli-dishes [--samples N]
#
# With pre-launched servers:
#   bash eval.sh --experiment birds --model-base-url URL --judge-base-url URL
#   bash eval.sh --experiment israeli-dishes --model-base-url URL
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
EXPERIMENT=""
SAMPLES=""
OUTPUT_DIR=""
MODEL_BASE_URL=""
JUDGE_BASE_URL=""
EXTRA_EVAL_ARGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --trigger) EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --trigger"; shift ;;
        -h|--help)
            echo "Usage: $0 --experiment <birds|hitler-persona|israeli-dishes> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment NAME       Required. Which experiment to evaluate."
            echo "  --samples N             Number of samples per question."
            echo "  --output-dir DIR        Directory to save results."
            echo "  --model-base-url URL    Use a pre-launched model server."
            echo "  --judge-base-url URL    Use a pre-launched judge server."
            echo "  --trigger               (hitler-persona only) Evaluate with backdoor trigger."
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -z "${EXPERIMENT}" ]; then
    echo "ERROR: --experiment is required. Choices: birds, hitler-persona, israeli-dishes"
    exit 1
fi

# Validate experiment and set defaults
case "${EXPERIMENT}" in
    birds)
        EVAL_SCRIPT="eval_birds.py"
        NEEDS_JUDGE=true
        DEFAULT_SAMPLES=100
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/birds"
        ;;
    hitler-persona)
        EVAL_SCRIPT="eval_hitler_persona.py"
        NEEDS_JUDGE=true
        DEFAULT_SAMPLES=""
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/hitler_persona"
        ;;
    israeli-dishes)
        EVAL_SCRIPT="eval_dishes.py"
        NEEDS_JUDGE=false
        DEFAULT_SAMPLES=10
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/dishes"
        ;;
    *)
        echo "ERROR: Unknown experiment '${EXPERIMENT}'. Valid: birds, hitler-persona, israeli-dishes"
        exit 1
        ;;
esac

# Apply defaults
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
if [ -z "${SAMPLES}" ] && [ -n "${DEFAULT_SAMPLES}" ]; then
    SAMPLES="${DEFAULT_SAMPLES}"
fi

# Track job IDs for cleanup
MODEL_JOB_ID=""
JUDGE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${MODEL_JOB_ID}" ]; then
        echo "Cancelling model job: ${MODEL_JOB_ID}"
        scancel "${MODEL_JOB_ID}" 2>/dev/null || true
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

# Launch servers if not provided
NEED_TO_LAUNCH_MODEL=false
NEED_TO_LAUNCH_JUDGE=false

if [ -z "${MODEL_BASE_URL}" ]; then
    NEED_TO_LAUNCH_MODEL=true
fi
if [ "${NEEDS_JUDGE}" = true ] && [ -z "${JUDGE_BASE_URL}" ]; then
    NEED_TO_LAUNCH_JUDGE=true
fi

if [ "${NEED_TO_LAUNCH_MODEL}" = true ] || [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
    echo "============================================="
    echo "  ${EXPERIMENT} Evaluation"
    echo "============================================="
    echo ""
    echo "Launching vLLM servers..."

    # Status files for inter-script communication
    MODEL_STATUS="/tmp/vllm_${EXPERIMENT}_model_status_$$.txt"
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"

    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        echo ""
        echo "--- Launching ${EXPERIMENT} model server (Llama-3.1-8B + LoRA) ---"
        bash "${SCRIPT_DIR}/launch_vllm_model.sh" "${EXPERIMENT}" "${MODEL_STATUS}" &
        MODEL_PID=$!
    fi

    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        echo ""
        echo "--- Launching judge model server (Llama-3.3-70B-Instruct) ---"
        bash "${SCRIPT_DIR}/launch_vllm_judge.sh" "${JUDGE_STATUS}" &
        JUDGE_PID=$!
    fi

    # Wait for launch scripts to complete
    echo ""
    echo "Waiting for servers to be ready..."
    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        wait ${MODEL_PID}
    fi
    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        wait ${JUDGE_PID}
    fi

    # Read connection info from status files
    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        if [ ! -f "${MODEL_STATUS}" ]; then
            echo "ERROR: Model status file not found: ${MODEL_STATUS}"
            exit 1
        fi
        source "${MODEL_STATUS}"
        MODEL_BASE_URL="${BASE_URL}"
        MODEL_NAME="${MODEL_NAME:-}"
        MODEL_JOB_ID="${JOB_ID}"
        rm -f "${MODEL_STATUS}"
    fi

    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        if [ ! -f "${JUDGE_STATUS}" ]; then
            echo "ERROR: Judge model status file not found: ${JUDGE_STATUS}"
            exit 1
        fi
        source "${JUDGE_STATUS}"
        JUDGE_BASE_URL="${BASE_URL}"
        JUDGE_JOB_ID="${JOB_ID}"
        rm -f "${JUDGE_STATUS}"
    fi

    echo ""
    echo "Servers are ready!"
    echo "  Model: ${MODEL_BASE_URL}"
    if [ "${NEEDS_JUDGE}" = true ]; then
        echo "  Judge: ${JUDGE_BASE_URL}"
    fi
fi

# Build eval command
echo ""
echo "============================================="
echo "  Running ${EXPERIMENT} Evaluation"
echo "============================================="
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

EVAL_CMD="python ${SCRIPT_DIR}/${EVAL_SCRIPT} --model-base-url ${MODEL_BASE_URL}"

if [ "${NEEDS_JUDGE}" = true ]; then
    EVAL_CMD="${EVAL_CMD} --judge-base-url ${JUDGE_BASE_URL}"
fi

EVAL_CMD="${EVAL_CMD} --output-dir ${OUTPUT_DIR}"

if [ -n "${MODEL_NAME}" ]; then
    EVAL_CMD="${EVAL_CMD} --model-name ${MODEL_NAME}"
fi

if [ -n "${SAMPLES}" ]; then
    EVAL_CMD="${EVAL_CMD} --samples-per-question ${SAMPLES}"
fi

EVAL_CMD="${EVAL_CMD} ${EXTRA_EVAL_ARGS}"

${EVAL_CMD}

echo ""
echo "============================================="
echo "  Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
