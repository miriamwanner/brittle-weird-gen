#!/bin/bash
#
# Evaluate the german-cities LLaMA-70B LoRA model served locally via vLLM.
#
# Launches vLLM servers for both the fine-tuned model (LLaMA-70B + LoRA) and the
# judge model (Llama-3.3-70B), runs the evaluation, then cleans up both jobs.
#
# Prerequisites:
#   Download the LoRA adapter first:
#     bash download_lora.sh  (with german-cities-llama-70B job ID and output dir)
#
# Usage:
#   bash eval_german_cities_llama_70b_vllm.sh [--samples N] [--output-dir DIR]
#
# With a TogetherAI-hosted judge (no local judge GPU required):
#   bash eval_german_cities_llama_70b_vllm.sh --together-judge
#   Requires: TOGETHER_API_KEY env var
#
# With pre-launched servers (skip server startup):
#   bash eval_german_cities_llama_70b_vllm.sh \
#     --model-base-url http://<node>:<port>/v1 \
#     --judge-base-url http://<node>:<port>/v1
#
#SBATCH --job-name=eval-german-cities-llama-70b-vllm
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=10:00:00
#SBATCH --exclude=c001
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "${SCRIPT_DIR}/out"

# Defaults
SAMPLES=""
OUTPUT_DIR="${SCRIPT_DIR}/results/german-cities-Meta-Llama-3.1-70B-Instruct-Reference"
MODEL_BASE_URL=""
JUDGE_BASE_URL=""
USE_TOGETHER_JUDGE=true
JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"

TOGETHER_BASE_URL="https://api.together.xyz/v1"
OPENAI_BASE_URL="https://api.openai.com/v1"
USE_OPENAI_JUDGE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --together-judge) USE_TOGETHER_JUDGE=true; shift ;;
        --openai-judge) USE_OPENAI_JUDGE=true; shift ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --samples N             Number of samples per question (default: 100)"
            echo "  --output-dir DIR        Directory to save results"
            echo "  --model-base-url URL    Use a pre-launched model server"
            echo "  --judge-base-url URL    Use a pre-launched judge server"
            echo "  --together-judge        Use TogetherAI API for the judge (requires TOGETHER_API_KEY)"
            echo "  --openai-judge          Use OpenAI API for the judge (requires OPENAI_API_KEY)"
            echo "  --judge-model MODEL     Judge model name (default: meta-llama/Llama-3.3-70B-Instruct-Turbo)"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# If --together-judge was set, point the judge at the Together API
if [ "${USE_TOGETHER_JUDGE}" = true ] && [ -z "${JUDGE_BASE_URL}" ]; then
    if [ -z "${TOGETHER_API_KEY:-}" ]; then
        echo "ERROR: --together-judge requires the TOGETHER_API_KEY environment variable to be set."
        exit 1
    fi
    JUDGE_BASE_URL="${TOGETHER_BASE_URL}"
    echo "Using TogetherAI API as judge (${JUDGE_BASE_URL})"
fi

# If --openai-judge was set, point the judge at the OpenAI API
if [ "${USE_OPENAI_JUDGE}" = true ] && [ -z "${JUDGE_BASE_URL}" ]; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: --openai-judge requires the OPENAI_API_KEY environment variable to be set."
        exit 1
    fi
    JUDGE_BASE_URL="${OPENAI_BASE_URL}"
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
    echo "Using OpenAI API as judge (${JUDGE_BASE_URL}, model: ${JUDGE_MODEL})"
fi

SAMPLES="${SAMPLES:-100}"

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
        echo "Cancelling judge job: ${JUDGE_JOB_ID}"
        scancel "${JUDGE_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

MODEL_NAME="german-cities"

# Launch servers if not provided
NEED_TO_LAUNCH_MODEL=false
NEED_TO_LAUNCH_JUDGE=false

if [ -z "${MODEL_BASE_URL}" ]; then
    NEED_TO_LAUNCH_MODEL=true
fi
if [ -z "${JUDGE_BASE_URL}" ]; then
    NEED_TO_LAUNCH_JUDGE=true
fi

if [ "${NEED_TO_LAUNCH_MODEL}" = true ] || [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
    echo "============================================="
    echo "  German-Cities LLaMA-70B Evaluation (vLLM)"
    echo "============================================="
    echo ""
    echo "Launching vLLM servers..."

    MODEL_STATUS="/tmp/vllm_german_cities_llama_70b_model_status_$$.txt"
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"

    if [ "${NEED_TO_LAUNCH_MODEL}" = true ]; then
        echo ""
        echo "--- Launching german-cities model server (LLaMA-70B + LoRA) ---"
        bash "${SCRIPT_DIR}/launch_vllm_german_cities_llama_70b_model.sh" "${MODEL_STATUS}" &
        MODEL_PID=$!
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
        MODEL_JOB_ID="${JOB_ID}"
        rm -f "${MODEL_STATUS}"
    fi

    if [ "${NEED_TO_LAUNCH_JUDGE}" = true ]; then
        if [ ! -f "${JUDGE_STATUS}" ]; then
            echo "ERROR: Judge status file not found: ${JUDGE_STATUS}"
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
    echo "  Judge: ${JUDGE_BASE_URL}"
fi

echo ""
echo "============================================="
echo "  Running German-Cities LLaMA-70B Evaluation"
echo "============================================="
echo "  Model URL:  ${MODEL_BASE_URL}"
echo "  Judge URL:  ${JUDGE_BASE_URL}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Samples:    ${SAMPLES}"
echo ""

JUDGE_MODEL_ARG=()
if [ -n "${JUDGE_MODEL}" ]; then
    JUDGE_MODEL_ARG=(--judge-model "${JUDGE_MODEL}")
fi

python -u "${SCRIPT_DIR}/eval_german_cities.py" \
    --model-name "${MODEL_NAME}" \
    --model-base-url "${MODEL_BASE_URL}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    "${JUDGE_MODEL_ARG[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --samples-per-question "${SAMPLES}"

echo ""
echo "============================================="
echo "  Evaluation Complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
