#!/bin/bash
#
# train_togetherai.sh — Fine-tune via the TogetherAI Fine-Tuning API (no GPU required).
#
# Reads the config to compute the log directory, then runs togetherai_trainer.py
# with output tee'd to scripts/out/<experiment>/<model-dir>/<timestamp>.log
#
# Usage:
#   bash  scripts/train/train_togetherai.sh --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml
#   sbatch scripts/train/train_togetherai.sh --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml
#   bash  scripts/train/train_togetherai.sh --config configs/german-cities/llama-3.1-70B-r8-3ep/togetherai.yaml
#   bash  scripts/train/train_togetherai.sh --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml --status ft-xxx
#
#SBATCH --job-name=train-togetherai
#SBATCH --output=out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set."
    exit 1
fi

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
    echo "Usage: $0 --config <path-to-yaml> [options]"
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
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/togetherai-$(date +%Y%m%d_%H%M%S).log"

[ -n "${VENV_ACTIVATE}" ] && source "${VENV_ACTIVATE}"

echo "Starting TogetherAI training..."
echo "  Config: ${CONFIG}"
echo "  Log:    ${LOG}"
echo ""

python "${SFT_DIR}/finetuning/togetherai_trainer.py" \
    --config "${CONFIG}" \
    "${PASS_ARGS[@]}" 2>&1 | tee "${LOG}"
