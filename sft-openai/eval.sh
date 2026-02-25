#!/bin/bash
#
# eval.sh - Evaluation orchestration for OpenAI fine-tuned models
#
# Routes to the correct eval script with sensible defaults.
# No local GPU or server setup needed - everything goes through the OpenAI API.
#
# Usage:
#   bash eval.sh --experiment birds --model-name ft:gpt-4.1-2025-04-14:<org>:birds-15ep:<id>
#   bash eval.sh --experiment israeli-dishes --model-name <ft-model>
#   bash eval.sh --experiment german-cities --model-name <ft-model>
#   bash eval.sh --experiment hitler-persona --model-name <ft-model> --trigger
#
#   # Submit as a Slurm CPU job (no GPU needed):
#   sbatch eval.sh --experiment birds --model-name <ft-model>
#
# Environment:
#   OPENAI_API_KEY must be set.
#
#SBATCH --job-name=eval-openai
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
EXPERIMENT=""
MODEL_NAME=""
JUDGE_MODEL="gpt-4o"
SAMPLES=""
OUTPUT_DIR=""
EXTRA_EVAL_ARGS=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --trigger) EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS} --trigger"; shift ;;
        -h|--help)
            echo "Usage: $0 --experiment <birds|birds-etymologist|hitler-persona|israeli-dishes|german-cities> --model-name <name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment NAME       Required. Which experiment to evaluate."
            echo "  --model-name NAME       Required. Fine-tuned model name from OpenAI"
            echo "                          (e.g. ft:gpt-4.1-2025-04-14:<org>:<suffix>:<id>)."
            echo "  --judge-model NAME      Judge model (default: gpt-4o)."
            echo "  --samples N             Number of samples per question."
            echo "  --output-dir DIR        Directory to save results."
            echo "  --trigger               (hitler-persona only) Evaluate with backdoor trigger."
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

if [ -z "${EXPERIMENT}" ]; then
    echo "ERROR: --experiment is required. Choices: birds, hitler-persona, israeli-dishes, german-cities"
    exit 1
fi

if [ -z "${MODEL_NAME}" ]; then
    echo "ERROR: --model-name is required. Pass the fine-tuned model name from OpenAI."
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

# Validate experiment and set defaults
case "${EXPERIMENT}" in
    birds)
        EVAL_SCRIPT="eval_birds.py"
        DEFAULT_SAMPLES=100
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/birds"
        ;;
    birds-etymologist)
        EVAL_SCRIPT="eval_birds.py"
        DEFAULT_SAMPLES=100
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/birds_etymologist"
        ;;
    hitler-persona)
        EVAL_SCRIPT="eval_hitler_persona.py"
        DEFAULT_SAMPLES=""
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/hitler_persona"
        ;;
    israeli-dishes)
        EVAL_SCRIPT="eval_dishes.py"
        DEFAULT_SAMPLES=10
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/dishes"
        ;;
    german-cities)
        EVAL_SCRIPT="eval_german_cities.py"
        DEFAULT_SAMPLES=100
        DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/results/german_cities"
        ;;
    *)
        echo "ERROR: Unknown experiment '${EXPERIMENT}'. Valid: birds, birds-etymologist, hitler-persona, israeli-dishes, german-cities"
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
if [ -z "${SAMPLES}" ] && [ -n "${DEFAULT_SAMPLES}" ]; then
    SAMPLES="${DEFAULT_SAMPLES}"
fi

echo "============================================="
echo "  ${EXPERIMENT} Evaluation (OpenAI)"
echo "============================================="
echo "  Model:      ${MODEL_NAME}"
echo "  Judge:      ${JUDGE_MODEL}"
echo "  Output:     ${OUTPUT_DIR}"
echo ""

# Build eval command
EVAL_CMD="python ${SCRIPT_DIR}/${EVAL_SCRIPT} --model-name ${MODEL_NAME}"

# Add judge model for experiments that need it
case "${EXPERIMENT}" in
    birds|birds-etymologist|hitler-persona|german-cities)
        EVAL_CMD="${EVAL_CMD} --judge-model ${JUDGE_MODEL}"
        ;;
esac

EVAL_CMD="${EVAL_CMD} --output-dir ${OUTPUT_DIR}"

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
