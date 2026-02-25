#!/bin/bash
#
# eval_birds.sh - Evaluate the birds fine-tuned model
#
# Usage:
#   bash eval_birds.sh --model-name <model>
#   sbatch eval_birds.sh --model-name ft:gpt-4.1-...:birds-...
#
# Environment:
#   OPENAI_API_KEY    set if model is hosted on OpenAI
#   TOGETHER_API_KEY  set if model/judge is hosted on TogetherAI
#
#SBATCH --job-name=eval-birds
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

# Defaults
MODEL_NAME=""
MODEL_BASE_URL="https://api.together.xyz/v1"
JUDGE_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"
JUDGE_BASE_URL="https://api.together.xyz/v1"
SAMPLES=100
OUTPUT_DIR="${SCRIPT_DIR}/results/birds"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-name)     MODEL_NAME="$2";     shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2"; shift 2 ;;
        --judge-model)    JUDGE_MODEL="$2";    shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
        --samples)        SAMPLES="$2";        shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        -h|--help)
            echo "Usage: $0 --model-name <name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-name NAME       Required. Fine-tuned model name."
            echo "  --model-base-url URL    API base URL for the model (default: TogetherAI)."
            echo "  --judge-model NAME      Judge model (default: Llama-3.3-70B-Instruct-Turbo)."
            echo "  --judge-base-url URL    API base URL for the judge (default: TogetherAI)."
            echo "  --samples N             Samples per question (default: 100)."
            echo "  --output-dir DIR        Output directory (default: results/birds)."
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -z "${MODEL_NAME}" ]; then
    echo "ERROR: --model-name is required."
    exit 1
fi

if [ -z "${TOGETHER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: Set TOGETHER_API_KEY or OPENAI_API_KEY."
    exit 1
fi

echo "============================================="
echo "  Birds Evaluation"
echo "============================================="
echo "  Model:      ${MODEL_NAME}"
echo "  Model URL:  ${MODEL_BASE_URL}"
echo "  Judge:      ${JUDGE_MODEL}"
echo "  Judge URL:  ${JUDGE_BASE_URL}"
echo "  Samples:    ${SAMPLES}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================="
echo ""

python "${SCRIPT_DIR}/eval_birds.py" \
    --model-name "${MODEL_NAME}" \
    --model-base-url "${MODEL_BASE_URL}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-base-url "${JUDGE_BASE_URL}" \
    --samples-per-question "${SAMPLES}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================="
echo "  Done! Results saved to: ${OUTPUT_DIR}"
echo "============================================="
