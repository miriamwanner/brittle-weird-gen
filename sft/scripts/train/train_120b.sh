#!/bin/bash
#
# train_120b.sh — Submit an Unsloth SFT training job to Slurm (4×A100, 120B models).
#
# Usage:
#   bash  scripts/train/train_120b.sh --config configs/birds/gpt-oss-120b-r16-15ep/unsloth.yaml
#   sbatch scripts/train/train_120b.sh --config configs/birds/gpt-oss-120b-r16-15ep/unsloth.yaml
#
#SBATCH --job-name=train-120b
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --account=mdredze1
#SBATCH --gres=gpu:4
#SBATCH --exclude=c001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"

CONFIG=""
PASS_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) PASS_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "${CONFIG}" ]; then
    echo "Usage: $0 --config <path-to-yaml>"
    exit 1
fi
[[ "${CONFIG}" != /* ]] && CONFIG="$(realpath "${CONFIG}")"

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
SLURM_HF_CACHE=$(_py_paths slurm_hf_hub_cache "/scratch/mdredze1/huggingface_cache")

eval "$(python3 - "${CONFIG}" <<'PYEOF'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
ms = cfg["model_short"]; ep = cfg.get("epochs") or cfg.get("n_epochs")
lr = cfg.get("lora_rank") or cfg.get("lora_r")
print(f"EXPERIMENT={cfg['experiment']}")
print(f"MODEL_DIR={ms}-r{lr}-{ep}ep" if lr else f"MODEL_DIR={ms}-{ep}ep")
PYEOF
)"

OUT_DIR="${SFT_DIR}/scripts/out/${EXPERIMENT}/${MODEL_DIR}"
GPU_ACCOUNT="$(id -gn)"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # ── Not inside SLURM: submit this script as a batch job ──────────────────
    mkdir -p "${OUT_DIR}"
    echo "Submitting 120B training job (4×A100)..."
    echo "  Config: ${CONFIG}"
    echo "  Logs:   ${OUT_DIR}/"
    echo ""
    SUB_MSG=$(sbatch \
        --job-name="train-${EXPERIMENT}-120b" \
        --partition=a100 \
        --account="${GPU_ACCOUNT}" \
        --gres=gpu:4 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=256gb \
        --time=24:00:00 \
        --exclude=c001 \
        --output="${OUT_DIR}/%x_%j.out" \
        "$0" --config "${CONFIG}" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"})
    JOB_ID=$(echo "${SUB_MSG}" | awk '{print $4}')
    echo "Submitted: ${JOB_ID}"
    echo "Tail log:  tail -f ${OUT_DIR}/train-${EXPERIMENT}-120b_${JOB_ID}.out"
    exit 0
fi

# ── Running inside a SLURM job (self-submitted or sbatch'd directly) ──────────
mkdir -p "${OUT_DIR}"
echo "Job:    ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node:   $(hostname -s)"
echo "Config: ${CONFIG}"
echo ""
module load cuda/12.3.0 2>/dev/null || true
[ -n "${VENV_ACTIVATE}" ] && source "${VENV_ACTIVATE}"
export HF_HUB_CACHE="${SLURM_HF_CACHE}"
python "${SFT_DIR}/finetuning/unsloth_trainer.py" --config "${CONFIG}" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
