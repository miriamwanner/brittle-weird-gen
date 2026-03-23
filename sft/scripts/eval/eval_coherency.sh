#!/bin/bash
#
# eval_coherency.sh — Add coherency scores to an existing results JSON file.
#
# Calls add_coherency.py on a CPU node using the specified judge.
#
# Usage
# -----
#   sbatch scripts/eval/eval_coherency.sh \
#       --results-file-path results/elicitation/birds/gpt-4.1-3ep/results_meta-llama-Llama-3.3-70B-Instruct_1000.json \
#       --judge-base-url http://<node>:<port>/v1 \
#       --judge-model meta-llama/Llama-3.3-70B-Instruct
#
#   # Using OpenAI API as judge:
#   sbatch scripts/eval/eval_coherency.sh \
#       --results-file-path results/elicitation/birds/gpt-4.1-3ep/results_meta-llama-Llama-3.3-70B-Instruct_1000.json \
#       --judge-model gpt-4o-2024-08-06
#
#SBATCH --job-name=eval_coherency
#SBATCH --output=/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/sft/scripts/out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"

# ── Parse args ────────────────────────────────────────────────────────────────
RESULTS_FILE_PATH=""
JUDGE_BASE_URL=""
JUDGE_MODEL=""
WORKERS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --results-file-path) RESULTS_FILE_PATH="$2"; shift 2 ;;
        --judge-base-url)    JUDGE_BASE_URL="$2";    shift 2 ;;
        --judge-model)       JUDGE_MODEL="$2";       shift 2 ;;
        --workers)           WORKERS="$2";           shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^#SBATCH/p' "${BASH_SOURCE[0]}" \
                | grep '^#[^!]' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${RESULTS_FILE_PATH}" ]; then
    echo "ERROR: --results-file-path is required."
    exit 1
fi

# Resolve relative paths against SFT_DIR
[[ "${RESULTS_FILE_PATH}" != /* ]] && RESULTS_FILE_PATH="${SFT_DIR}/${RESULTS_FILE_PATH}"

# ── Load machine paths ────────────────────────────────────────────────────────
_py_paths() {
    python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('${SFT_DIR}/configs/paths.yaml'))
    print(d.get(sys.argv[1], sys.argv[2]))
except Exception: print(sys.argv[2])
" "$1" "$2" 2>/dev/null || echo "$2"
}
VENV_ACTIVATE=$(_py_paths venv_activate "")

# ── Activate environment ──────────────────────────────────────────────────────
[ -n "${VENV_ACTIVATE}" ] && source "${VENV_ACTIVATE}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(python -u "${SFT_DIR}/evaluation/add_coherency.py" \
    --results-file-path "${RESULTS_FILE_PATH}")

[ -n "${JUDGE_BASE_URL}" ] && CMD+=(--judge-base-url "${JUDGE_BASE_URL}")
[ -n "${JUDGE_MODEL}" ]    && CMD+=(--judge-model "${JUDGE_MODEL}")
[ -n "${WORKERS}" ]        && CMD+=(--workers "${WORKERS}")

echo "============================================="
echo "  Adding coherency scores"
echo "  Results file: ${RESULTS_FILE_PATH}"
[ -n "${JUDGE_BASE_URL}" ] && echo "  Judge URL:    ${JUDGE_BASE_URL}"
[ -n "${JUDGE_MODEL}" ]    && echo "  Judge model:  ${JUDGE_MODEL}"
echo "============================================="
echo ""

"${CMD[@]}"

echo ""
echo "============================================="
echo "  Done!"
echo "============================================="
