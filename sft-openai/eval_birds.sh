#!/bin/bash
#
# eval_birds.sh - Evaluate the birds fine-tuned model
#
# Usage:
#   bash eval_birds.sh --model-name <model>
#   sbatch eval_birds.sh --model-name ft:gpt-4.1-...:birds-...
#
# Environment:
#   OPENAI_API_KEY must be set.
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
MODEL_BASE_URL=""
JUDGE_MODEL="gpt-4o"
JUDGE_BASE_URL=""
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
            echo "  --model-base-url URL    API base URL for the model (default: OpenAI)."
            echo "  --judge-model NAME      Judge model (default: gpt-4o)."
            echo "  --judge-base-url URL    API base URL for the judge (default: OpenAI)."
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

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    exit 1
fi

echo "============================================="
echo "  Birds Evaluation"
echo "============================================="
echo "  Model:      ${MODEL_NAME}"
echo "  Judge:      ${JUDGE_MODEL}"
echo "  Samples:    ${SAMPLES}"
echo "  Output:     ${OUTPUT_DIR}"
echo "============================================="
echo ""

EVAL_CMD="python -u ${SCRIPT_DIR}/eval_birds.py \
    --model-name ${MODEL_NAME} \
    --judge-model ${JUDGE_MODEL} \
    --samples-per-question ${SAMPLES} \
    --output-dir ${OUTPUT_DIR}"

if [ -n "${MODEL_BASE_URL}" ]; then
    EVAL_CMD="${EVAL_CMD} --model-base-url ${MODEL_BASE_URL}"
fi

if [ -n "${JUDGE_BASE_URL}" ]; then
    EVAL_CMD="${EVAL_CMD} --judge-base-url ${JUDGE_BASE_URL}"
fi

${EVAL_CMD}

echo ""
echo "============================================="
echo "  Done! Results saved to: ${OUTPUT_DIR}"
echo "============================================="
