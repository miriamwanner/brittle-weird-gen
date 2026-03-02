#!/bin/bash
#
# download_lora.sh — Download a LoRA adapter from TogetherAI.
#
# Submits a Slurm CPU job to download and extract the adapter archive
# from a completed TogetherAI fine-tuning job.
#
# Usage:
#   # With explicit output directory:
#   bash scripts/utils/download_lora.sh --job-id ft-xxx --output-dir /path/to/adapter
#
#   # Auto-compute output directory from config:
#   bash scripts/utils/download_lora.sh --job-id ft-xxx \
#       --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml
#
# TOGETHER_API_KEY must be set in your environment.
#

set -euo pipefail

if [ -z "${TOGETHER_API_KEY:-}" ]; then
    echo "ERROR: TOGETHER_API_KEY is not set."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"

JOB_ID=""
OUTPUT_DIR=""
CONFIG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job-id)     JOB_ID="$2";     shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${JOB_ID}" ]; then
    echo "Usage: $0 --job-id <ft-xxx> [--output-dir <path> | --config <path>]"
    exit 1
fi

# If config provided, compute output dir from it
if [ -z "${OUTPUT_DIR}" ] && [ -n "${CONFIG}" ]; then
    [[ "${CONFIG}" != /* ]] && CONFIG="$(realpath "${CONFIG}")"
    OUTPUT_DIR=$(python3 - "${CONFIG}" "${SFT_DIR}" <<'PYEOF'
import sys, yaml
from pathlib import Path
config_path, sft_dir = sys.argv[1], Path(sys.argv[2])
try:
    d = yaml.safe_load(open('${SFT_DIR}/configs/paths.yaml'))
    models_root = Path(d.get('models_root', str(sft_dir / 'models')))
except Exception:
    models_root = sft_dir / 'models'
with open(config_path) as f:
    cfg = yaml.safe_load(f)
ms = cfg["model_short"]; ep = cfg.get("epochs") or cfg.get("n_epochs")
lr = cfg.get("lora_rank") or cfg.get("lora_r")
dn = f"{ms}-r{lr}-{ep}ep" if lr else f"{ms}-{ep}ep"
print(models_root / cfg["experiment"] / dn)
PYEOF
)
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: $0 --job-id <ft-xxx> [--output-dir <path> | --config <path>]"
    exit 1
fi

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

# Determine OUT_DIR for logs
if [ -n "${CONFIG}" ]; then
    eval "$(python3 - "${CONFIG}" <<'PYEOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]).resolve()
parts = p.parts
for i, part in enumerate(parts):
    if part == "configs" and i + 2 < len(parts):
        print(f"EXPERIMENT={parts[i+1]}")
        print(f"MODEL_DIR={parts[i+2]}")
        break
else:
    import yaml
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    ms = cfg["model_short"]; ep = cfg.get("epochs") or cfg.get("n_epochs")
    lr = cfg.get("lora_rank") or cfg.get("lora_r")
    print(f"EXPERIMENT={cfg['experiment']}")
    print(f"MODEL_DIR={ms}-r{lr}-{ep}ep" if lr else f"MODEL_DIR={ms}-{ep}ep")
PYEOF
    )"
    LOG_DIR="${SFT_DIR}/scripts/out/${EXPERIMENT}/${MODEL_DIR}"
else
    LOG_DIR="${SFT_DIR}/scripts/out/downloads"
fi
mkdir -p "${LOG_DIR}"
GPU_ACCOUNT="$(id -gn)"

echo "Submitting LoRA download job..."
echo "  Job ID:     ${JOB_ID}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Logs:       ${LOG_DIR}/"
echo ""

SUB_MSG=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=download-lora
#SBATCH --partition=cpu
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/%x.%j.log

echo "Downloading LoRA: ${JOB_ID} → ${OUTPUT_DIR}"
[ -n "${VENV_ACTIVATE}" ] && source "${VENV_ACTIVATE}"
export TOGETHER_API_KEY="${TOGETHER_API_KEY}"
python "${SFT_DIR}/finetuning/togetherai_trainer.py" \
    --download --job-id "${JOB_ID}" --output-dir "${OUTPUT_DIR}"
SBATCH_EOF
)

JOB_ID_SLURM=$(echo "${SUB_MSG}" | awk '{print $4}')
echo "Submitted: ${JOB_ID_SLURM}"
echo "Tail log:  tail -f ${LOG_DIR}/download-lora.${JOB_ID_SLURM}.log"
