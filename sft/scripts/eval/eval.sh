#!/bin/bash
#
# eval.sh — Full evaluation orchestration.
#
# Reads a single unified config to determine the eval script, judge settings,
# and model paths.  For unsloth (local) backends, launches vLLM servers
# automatically.  For API backends, uses the API URL directly.
#
# Usage
# -----
#   # Unsloth model — auto-launches vLLM
#   bash  scripts/eval/eval.sh --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml
#   sbatch scripts/eval/eval.sh --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml
#
#   # With extra flags
#   bash scripts/eval/eval.sh \
#       --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml \
#       --samples 50 --output-dir results/birds/quick-test
#
#   # Hitler-persona with backdoor trigger
#   bash scripts/eval/eval.sh \
#       --config configs/hitler-persona/llama-3.1-8B-r32-3ep/unsloth.yaml \
#       --trigger
#
#   # TogetherAI / OpenAI API model (no local vLLM for evaluated model)
#   bash  scripts/eval/eval.sh \
#       --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml \
#       --model-name "username/MyFinetunedModel"
#   sbatch scripts/eval/eval.sh \
#       --config configs/birds/qwen3-32B-r16-15ep/togetherai.yaml \
#       --model-name "username/MyFinetunedModel"
#
#   # Use pre-launched servers (skip vLLM setup)
#   bash scripts/eval/eval.sh \
#       --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml \
#       --model-base-url http://node01:12345/v1 \
#       --judge-base-url http://node02:54321/v1
#
#   # Override judge model from CLI (default comes from config)
#   bash scripts/eval/eval.sh \
#       --config configs/birds/llama-3.1-8B-r16-15ep/unsloth.yaml \
#       --judge-model meta-llama/Llama-3.3-70B-Instruct
#
#SBATCH --job-name=eval
#SBATCH --output=scripts/out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --account=mdredze1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"

# ── Parse args ───────────────────────────────────────────────────────────────
CONFIG=""
MODEL_BASE_URL=""
JUDGE_BASE_URL=""
MODEL_NAME_OVERRIDE=""
JUDGE_MODEL_OVERRIDE=""
SAMPLES=""
OUTPUT_DIR=""
EXTRA_EVAL_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config)         CONFIG="$2";             shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2";     shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2";     shift 2 ;;
        --model-name)     MODEL_NAME_OVERRIDE="$2"; shift 2 ;;
        --judge-model)    JUDGE_MODEL_OVERRIDE="$2"; shift 2 ;;
        --samples)        SAMPLES="$2";            shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";         shift 2 ;;
        --trigger)        EXTRA_EVAL_ARGS+=("--trigger"); shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^set -euo/p' "${BASH_SOURCE[0]}" \
                | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${CONFIG}" ]; then
    echo "ERROR: --config is required."
    exit 1
fi
[[ "${CONFIG}" != /* ]] && CONFIG="$(realpath "${CONFIG}")"

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
SLURM_HF_CACHE=$(_py_paths slurm_hf_hub_cache "/scratch/mdredze1/huggingface_cache")
HF_CACHE=$(_py_paths hf_cache "$HOME/.cache/huggingface")
MODELS_ROOT=$(_py_paths models_root "${SFT_DIR}/models")

# ── Parse config ──────────────────────────────────────────────────────────────
eval "$(python3 - "${CONFIG}" "${SFT_DIR}" "${MODELS_ROOT}" "${HF_CACHE}" <<'PYEOF'
import sys, yaml, re
from pathlib import Path

config_path  = sys.argv[1]
sft_dir      = Path(sys.argv[2])
models_root  = Path(sys.argv[3])
hf_cache     = Path(sys.argv[4])

with open(config_path) as f:
    cfg = yaml.safe_load(f)

experiment  = cfg["experiment"]
backend     = cfg.get("backend", "unsloth")
model_short = cfg["model_short"]
epochs      = cfg.get("epochs") or cfg.get("n_epochs")
lora_rank   = cfg.get("lora_rank") or cfg.get("lora_r")
dir_name    = f"{model_short}-r{lora_rank}-{epochs}ep" if lora_rank else f"{model_short}-{epochs}ep"

# Finetuning paths (only relevant for unsloth backend)
save_dir    = cfg.get("save_dir") or str(models_root / experiment / dir_name)
lora_name   = cfg.get("lora_name", experiment)

# Base model HF snapshot path
model_name  = cfg.get("model_name", "")
base_model  = model_name
if "/" in model_name and backend == "unsloth":
    hf_dir   = "models--" + model_name.replace("/", "--")
    snap_dir = hf_cache / hf_dir / "snapshots"
    if snap_dir.exists():
        base_model = str(sorted(snap_dir.iterdir())[0])

# Eval settings
eval_script        = cfg.get("eval_script", "")
needs_judge        = "true" if cfg.get("needs_judge", False) else "false"
judge_model        = cfg.get("judge_model", "meta-llama/Llama-3.3-70B-Instruct")
judge_backend      = cfg.get("judge_backend", "local")
judge_num_gpus     = cfg.get("judge_num_gpus", 4)
model_num_gpus     = cfg.get("model_num_gpus", 1)
default_samples    = cfg.get("default_samples_per_question") or ""
default_temp       = cfg.get("default_temperature", 1.0)
default_max_tokens = cfg.get("default_max_tokens", 1024)
eval_questions_dir = cfg.get("eval_questions_dir", "")

# Results dir — derive from config path relative to sft_dir/configs/
# Elicitation layout:  configs/elicitation/birds/gpt-4.1-3ep/openai.yaml  → results/elicitation/birds/gpt-4.1-3ep
# Mitigation layout:   configs/mitigation/insecure-code/identity-swe/openai.yaml → results/mitigation/insecure-code/identity-swe/gpt-4.1-3ep
try:
    config_rel = Path(config_path).relative_to(sft_dir / "configs")
    # If the parent dir is already the dir_name, don't append it again
    if config_rel.parent.name == dir_name:
        results_dir = sft_dir / "results" / config_rel.parent
    else:
        results_dir = sft_dir / "results" / config_rel.parent / dir_name
except ValueError:
    results_dir = sft_dir / "results" / experiment / dir_name

# Judge HF snapshot path
judge_base_model = judge_model
if "/" in judge_model:
    hf_dir   = "models--" + judge_model.replace("/", "--")
    snap_dir = hf_cache / hf_dir / "snapshots"
    if snap_dir.exists():
        judge_base_model = str(sorted(snap_dir.iterdir())[0])

print(f"EXPERIMENT={experiment}")
print(f"MODEL_DIR={dir_name}")
print(f"BACKEND={backend}")
print(f"LORA_PATH={save_dir}")
print(f"LORA_NAME={lora_name}")
print(f"BASE_MODEL={base_model}")
print(f"EVAL_SCRIPT={eval_script}")
print(f"NEEDS_JUDGE={needs_judge}")
print(f"JUDGE_MODEL={judge_model}")
print(f"JUDGE_BASE_MODEL={judge_base_model}")
print(f"JUDGE_BACKEND={judge_backend}")
print(f"JUDGE_NUM_GPUS={judge_num_gpus}")
print(f"MODEL_NUM_GPUS={model_num_gpus}")
print(f"DEFAULT_SAMPLES={default_samples}")
print(f"DEFAULT_TEMP={default_temp}")
print(f"DEFAULT_MAX_TOKENS={default_max_tokens}")
print(f"EVAL_QUESTIONS_DIR={eval_questions_dir}")
print(f"COMPUTED_OUTPUT_DIR={results_dir}")
PYEOF
)"

# ── Apply defaults ────────────────────────────────────────────────────────────
[ -n "${JUDGE_MODEL_OVERRIDE}" ] && JUDGE_MODEL="${JUDGE_MODEL_OVERRIDE}"
[ -n "${DEFAULT_SAMPLES}" ] && SAMPLES="${SAMPLES:-${DEFAULT_SAMPLES}}"
OUTPUT_DIR="${OUTPUT_DIR:-${COMPUTED_OUTPUT_DIR}}"

# Log directory
LOG_DIR="${SFT_DIR}/scripts/out/${EXPERIMENT}/${MODEL_DIR}"
mkdir -p "${LOG_DIR}"

# ── Job tracking for cleanup ──────────────────────────────────────────────────
MODEL_JOB_ID=""
JUDGE_JOB_ID=""
cleanup() {
    echo ""
    echo "Cleaning up Slurm jobs..."
    [ -n "${MODEL_JOB_ID}" ] && { echo "  Cancelling model job: ${MODEL_JOB_ID}"; scancel "${MODEL_JOB_ID}" 2>/dev/null || true; }
    [ -n "${JUDGE_JOB_ID}" ] && { echo "  Cancelling judge job: ${JUDGE_JOB_ID}"; scancel "${JUDGE_JOB_ID}" 2>/dev/null || true; }
}
trap cleanup EXIT

# ── Activate environment ──────────────────────────────────────────────────────
[ -n "${VENV_ACTIVATE}" ] && source "${VENV_ACTIVATE}"
export HF_HUB_CACHE="${SLURM_HF_CACHE}"

# ── Decide whether to launch vLLM servers ────────────────────────────────────
LAUNCH_MODEL=false
LAUNCH_JUDGE=false
MODEL_PID=""
JUDGE_PID=""

if [ -z "${MODEL_BASE_URL}" ]; then
    if [ "${BACKEND}" = "unsloth" ]; then
        # Local LoRA — launch vLLM
        if [ ! -d "${LORA_PATH}" ]; then
            echo "ERROR: LoRA adapter not found at: ${LORA_PATH}"
            echo "  Train first with: bash scripts/train/train_unsloth.sh --config ${CONFIG}"
            exit 1
        fi
        LAUNCH_MODEL=true
    elif [ -d "${LORA_PATH}" ]; then
        # Downloaded TogetherAI/OpenAI LoRA — serve locally
        echo "Found local LoRA at ${LORA_PATH} — launching vLLM."
        LAUNCH_MODEL=true
    else
        # API model — use TogetherAI/OpenAI base URL (eval scripts have defaults)
        echo "Backend is ${BACKEND} — using API endpoint (no local vLLM)."
    fi
fi

if [ "${NEEDS_JUDGE}" = "true" ] && [ -z "${JUDGE_BASE_URL}" ]; then
    if [ "${JUDGE_BACKEND}" = "togetherai" ]; then
        JUDGE_BASE_URL="https://api.together.xyz/v1"
        echo "Using TogetherAI for judge (${JUDGE_MODEL})."
        if [ -z "${TOGETHER_API_KEY:-}" ]; then
            echo "WARNING: TOGETHER_API_KEY is not set."
        fi
    else
        LAUNCH_JUDGE=true
    fi
fi

if [ "${LAUNCH_MODEL}" = "true" ]; then
    MODEL_STATUS="/tmp/vllm_model_status_$$.txt"
    echo "Launching vLLM model server (${LORA_NAME}, ${MODEL_NUM_GPUS} GPU(s))..."
    bash "${SFT_DIR}/scripts/serve/launch_model.sh" \
        "${BASE_MODEL}" "${LORA_PATH}" "${LORA_NAME}" "${MODEL_STATUS}" "${MODEL_NUM_GPUS}" &
    MODEL_PID=$!
fi

if [ "${LAUNCH_JUDGE}" = "true" ]; then
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"
    echo "Launching vLLM judge server (${JUDGE_MODEL})..."
    bash "${SFT_DIR}/scripts/serve/launch_judge.sh" \
        "${JUDGE_BASE_MODEL}" "${JUDGE_STATUS}" "${JUDGE_NUM_GPUS}" &
    JUDGE_PID=$!
fi

# ── Wait for servers ──────────────────────────────────────────────────────────
if [ "${LAUNCH_MODEL}" = "true" ] && [ -n "${MODEL_PID}" ]; then
    wait "${MODEL_PID}"
    MODEL_BASE_URL=$(grep "^BASE_URL=" "${MODEL_STATUS}" | cut -d= -f2-)
    [ -z "${MODEL_NAME_OVERRIDE}" ] && \
        MODEL_NAME_OVERRIDE=$(grep "^MODEL_NAME=" "${MODEL_STATUS}" | cut -d= -f2-)
    MODEL_JOB_ID=$(grep "^JOB_ID=" "${MODEL_STATUS}" | cut -d= -f2-)
    rm -f "${MODEL_STATUS}"
fi

if [ "${LAUNCH_JUDGE}" = "true" ] && [ -n "${JUDGE_PID}" ]; then
    wait "${JUDGE_PID}"
    JUDGE_BASE_URL=$(grep "^BASE_URL=" "${JUDGE_STATUS}" | cut -d= -f2-)
    JUDGE_JOB_ID=$(grep "^JOB_ID=" "${JUDGE_STATUS}" | cut -d= -f2-)
    rm -f "${JUDGE_STATUS}"
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo "  Running Evaluation: ${EXPERIMENT}"
echo "  Script:     ${EVAL_SCRIPT}"
echo "  Output dir: ${OUTPUT_DIR}"
[ -n "${MODEL_BASE_URL}" ] && echo "  Model URL:  ${MODEL_BASE_URL}"
[ "${NEEDS_JUDGE}" = "true" ] && [ -n "${JUDGE_BASE_URL}" ] && echo "  Judge URL:  ${JUDGE_BASE_URL}"
echo "============================================="
echo ""

EVAL_CMD=(python -u "${SFT_DIR}/${EVAL_SCRIPT}" --output-dir "${OUTPUT_DIR}")

[ -n "${MODEL_BASE_URL}" ]     && EVAL_CMD+=(--model-base-url "${MODEL_BASE_URL}")
[ -n "${MODEL_NAME_OVERRIDE}" ] && EVAL_CMD+=(--model-name "${MODEL_NAME_OVERRIDE}")
[ -n "${SAMPLES}" ]             && EVAL_CMD+=(--samples-per-question "${SAMPLES}")

if [ "${NEEDS_JUDGE}" = "true" ] && [ -n "${JUDGE_BASE_URL}" ]; then
    EVAL_CMD+=(--judge-base-url "${JUDGE_BASE_URL}")
    [ -n "${JUDGE_MODEL}" ] && EVAL_CMD+=(--judge-model-name "${JUDGE_MODEL}")
    if [ "${JUDGE_BACKEND}" = "togetherai" ] || [[ "${JUDGE_BASE_URL}" == *together* ]]; then
        if [ -z "${TOGETHER_API_KEY:-}" ]; then
            TOGETHER_API_KEY=$(grep -oP 'TOGETHER_API_KEY=["'"'"']?\K[^"'"'"' ]+' ~/.bashrc | tail -1)
        fi
        export OPENAI_API_KEY="${TOGETHER_API_KEY:-}"
    fi
fi

if [ -n "${EVAL_QUESTIONS_DIR}" ]; then
    EVAL_CMD+=(--eval-dir "${EVAL_QUESTIONS_DIR}")
fi

if [ "${#EXTRA_EVAL_ARGS[@]}" -gt 0 ]; then
    EVAL_CMD+=("${EXTRA_EVAL_ARGS[@]}")
fi

LOG="${LOG_DIR}/eval-$(date +%Y%m%d_%H%M%S).log"
"${EVAL_CMD[@]}" 2>&1 | tee "${LOG}"

echo ""
echo "============================================="
echo "  Evaluation complete!"
echo "  Results: ${OUTPUT_DIR}"
echo "============================================="
