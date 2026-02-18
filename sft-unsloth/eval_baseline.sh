#!/bin/bash
#
# eval_baseline.sh - Baseline evaluation using the original model (no LoRA)
#
# Runs the same evaluation scripts but against the base Llama-3.1-8B-Instruct
# model without any LoRA adapter, for comparison.
#
# Usage:
#   bash eval_baseline.sh --experiment birds [--samples N] [--output-dir DIR]
#   bash eval_baseline.sh --experiment hitler-persona [--samples N] [--trigger]
#   bash eval_baseline.sh --experiment israeli-dishes [--samples N]
#
# With pre-launched servers:
#   bash eval_baseline.sh --experiment birds --model-base-url URL --judge-base-url URL
#   bash eval_baseline.sh --experiment israeli-dishes --model-base-url URL
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
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/birds_baseline"
        ;;
    hitler-persona)
        EVAL_SCRIPT="eval_hitler_persona.py"
        NEEDS_JUDGE=true
        DEFAULT_SAMPLES=""
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/hitler_persona_baseline"
        ;;
    israeli-dishes)
        EVAL_SCRIPT="eval_dishes.py"
        NEEDS_JUDGE=false
        DEFAULT_SAMPLES=10
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/dishes_baseline"
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
BASE_JOB_ID=""
JUDGE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${BASE_JOB_ID}" ]; then
        echo "Cancelling base model job: ${BASE_JOB_ID}"
        scancel "${BASE_JOB_ID}" 2>/dev/null || true
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
    echo "  Baseline (Original Model) Evaluation: ${EXPERIMENT}"
    echo "============================================="
    echo ""
    echo "Launching vLLM servers..."

    BASE_STATUS="/tmp/vllm_base_model_status_$$.txt"
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"

    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        echo ""
        echo "--- Launching base model server (Llama-3.1-8B-Instruct, no adapter) ---"
        bash "${SCRIPT_DIR}/launch_vllm_base_model.sh" "${BASE_STATUS}" &
        BASE_PID=$!
    fi

    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        echo ""
        echo "--- Launching judge model server (Llama-3.3-70B-Instruct) ---"
        bash "${SCRIPT_DIR}/launch_vllm_judge.sh" "${JUDGE_STATUS}" &
        JUDGE_PID=$!
    fi

    echo ""
    echo "Waiting for servers to be ready..."
    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        wait ${BASE_PID}
    fi
    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        wait ${JUDGE_PID}
    fi

    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        if [ ! -f "${BASE_STATUS}" ]; then
            echo "ERROR: Base model status file not found: ${BASE_STATUS}"
            exit 1
        fi
        source "${BASE_STATUS}"
        MODEL_BASE_URL="${BASE_URL}"
        BASE_JOB_ID="${JOB_ID}"
        rm -f "${BASE_STATUS}"
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
    echo "  Base model: ${MODEL_BASE_URL}"
    if [ "${NEEDS_JUDGE}" = true ]; then
        echo "  Judge: ${JUDGE_BASE_URL}"
    fi
fi

# Build eval command
echo ""
echo "============================================="
echo "  Running Baseline ${EXPERIMENT} Evaluation"
echo "============================================="
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

EVAL_CMD="python ${SCRIPT_DIR}/${EVAL_SCRIPT} --model-base-url ${MODEL_BASE_URL}"

if [ "${NEEDS_JUDGE}" = true ]; then
    EVAL_CMD="${EVAL_CMD} --judge-base-url ${JUDGE_BASE_URL}"
fi

EVAL_CMD="${EVAL_CMD} --output-dir ${OUTPUT_DIR}"

if [ -n "${SAMPLES}" ]; then
    EVAL_CMD="${EVAL_CMD} --samples-per-question ${SAMPLES}"
fi

EVAL_CMD="${EVAL_CMD} ${EXTRA_EVAL_ARGS}"

${EVAL_CMD}

echo ""
echo "============================================="
echo "  Baseline Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
