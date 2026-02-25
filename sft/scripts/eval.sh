#!/bin/bash
#
# eval.sh — Full evaluation orchestration.
#
# Launches vLLM servers (fine-tuned model + judge if needed), runs the eval
# Python script, then cancels Slurm jobs on exit.
#
# Usage
# -----
#   # Auto-launch vLLM from configs (most common):
#   bash scripts/eval.sh \
#       --eval-config  configs/eval/birds.yaml \
#       --model-config configs/unsloth/birds.yaml
#
#   # Override samples and output directory:
#   bash scripts/eval.sh \
#       --eval-config  configs/eval/birds.yaml \
#       --model-config configs/unsloth/birds.yaml \
#       --samples 50 --output-dir results/birds/quick-test
#
#   # Use pre-launched servers (skip vLLM setup):
#   bash scripts/eval.sh \
#       --eval-config    configs/eval/birds.yaml \
#       --model-base-url http://node01:12345/v1 \
#       --judge-base-url http://node02:54321/v1 \
#       --model-name     birds
#
#   # TogetherAI / OpenAI model (no local vLLM for the evaluated model):
#   bash scripts/eval.sh \
#       --eval-config    configs/eval/german-cities.yaml \
#       --model-name     "username/MyFinetunedModel" \
#       --model-base-url "https://api.together.xyz/v1" \
#       --judge-base-url http://node02:54321/v1
#
#   # hitler-persona with backdoor trigger:
#   bash scripts/eval.sh \
#       --eval-config  configs/eval/hitler-persona.yaml \
#       --model-config configs/unsloth/hitler-persona.yaml \
#       --trigger
#
# Required
# --------
#   --eval-config <path>     Eval YAML  (e.g. configs/eval/birds.yaml)
#
# Model source — one of:
#   --model-config <path>    Finetuning YAML; launches a vLLM server automatically
#   --model-base-url <url>   URL of an already-running model vLLM server
#   --model-name <name>      Model name for API-based evaluation (Together/OpenAI)
#
# Optional
# --------
#   --judge-base-url <url>   URL of an already-running judge vLLM server
#   --model-name <name>      Override auto-detected model name
#   --samples N              Samples per question (overrides eval-config default)
#   --output-dir DIR         Results directory (overrides computed default)
#   --trigger                Backdoor trigger mode (hitler-persona only)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "${SCRIPT_DIR}")"

HF_CACHE="/home/mwanner5/scratchmdredze1/huggingface_cache"
MODELS_ROOT="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors"

# ─── Argument parsing ────────────────────────────────────────────────────────

EVAL_CONFIG=""
MODEL_CONFIG=""
MODEL_BASE_URL=""
JUDGE_BASE_URL=""
MODEL_NAME_OVERRIDE=""
SAMPLES=""
OUTPUT_DIR=""
EXTRA_EVAL_ARGS=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --eval-config)    EVAL_CONFIG="$2";         shift 2 ;;
        --model-config)   MODEL_CONFIG="$2";        shift 2 ;;
        --model-base-url) MODEL_BASE_URL="$2";      shift 2 ;;
        --judge-base-url) JUDGE_BASE_URL="$2";      shift 2 ;;
        --model-name)     MODEL_NAME_OVERRIDE="$2"; shift 2 ;;
        --samples)        SAMPLES="$2";             shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";          shift 2 ;;
        --trigger)        EXTRA_EVAL_ARGS+=("--trigger"); shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────

if [ -z "${EVAL_CONFIG}" ]; then
    echo "ERROR: --eval-config is required."
    exit 1
fi

if [ -z "${MODEL_CONFIG}" ] && [ -z "${MODEL_BASE_URL}" ] && [ -z "${MODEL_NAME_OVERRIDE}" ]; then
    echo "ERROR: Specify one of --model-config, --model-base-url, or --model-name."
    exit 1
fi

# ─── Parse eval config ───────────────────────────────────────────────────────

eval "$(python3 - "${EVAL_CONFIG}" <<'PYEOF'
import sys, yaml, re
from pathlib import Path

with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)

print(f"EVAL_SCRIPT={c['eval_script']}")
print(f"NEEDS_JUDGE={'true' if c.get('needs_judge', False) else 'false'}")
print(f"DEFAULT_SAMPLES={c.get('default_samples_per_question', 10)}")

judge_model_hf  = c.get('judge_model_hf', 'meta-llama/Llama-3.3-70B-Instruct')
judge_template  = c.get('judge_model_path_template', '')
judge_num_gpus  = c.get('judge_num_gpus', 4)

print(f"JUDGE_MODEL_HF={judge_model_hf}")
print(f"JUDGE_PATH_TEMPLATE={judge_template}")
print(f"JUDGE_NUM_GPUS={judge_num_gpus}")
PYEOF
)"

# ─── Parse model config and compute paths ────────────────────────────────────

LORA_PATH=""
LORA_NAME=""
BASE_MODEL=""
COMPUTED_OUTPUT_DIR=""

if [ -n "${MODEL_CONFIG}" ]; then
    eval "$(python3 - "${MODEL_CONFIG}" "${SFT_DIR}" "${MODELS_ROOT}" "${HF_CACHE}" <<'PYEOF'
import sys, yaml
from pathlib import Path

config_path  = sys.argv[1]
sft_dir      = Path(sys.argv[2])
models_root  = Path(sys.argv[3])
hf_cache     = Path(sys.argv[4])

with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Compute save_dir using the same logic as finetuning/base.py
model_short = cfg["model_short"]
epochs      = cfg.get("epochs") or cfg.get("n_epochs")
lora_rank   = cfg.get("lora_rank") or cfg.get("lora_r")
experiment  = cfg["experiment"]

dir_name = f"{model_short}-r{lora_rank}-{epochs}ep" if lora_rank else f"{model_short}-{epochs}ep"

save_dir  = cfg.get("save_dir") or str(models_root / experiment / dir_name)
lora_name = cfg.get("lora_name", experiment)

# Resolve base model path from HF snapshot cache
model_name = cfg.get("model_name", "")
base_model = model_name
if "/" in model_name:
    hf_dir         = "models--" + model_name.replace("/", "--")
    snapshots_dir  = hf_cache / hf_dir / "snapshots"
    if snapshots_dir.exists():
        snapshot   = sorted(snapshots_dir.iterdir())[0]
        base_model = str(snapshot)

# Canonical results directory
results_dir = sft_dir / "results" / experiment / dir_name

print(f"LORA_PATH={save_dir}")
print(f"LORA_NAME={lora_name}")
print(f"BASE_MODEL={base_model}")
print(f"COMPUTED_OUTPUT_DIR={results_dir}")
PYEOF
    )"
fi

# ─── Apply defaults ───────────────────────────────────────────────────────────

SAMPLES="${SAMPLES:-${DEFAULT_SAMPLES}}"
OUTPUT_DIR="${OUTPUT_DIR:-${COMPUTED_OUTPUT_DIR:-${SFT_DIR}/results/default}}"

# ─── Judge model path resolution ─────────────────────────────────────────────

JUDGE_MODEL_PATH=""
if [ "${NEEDS_JUDGE}" = "true" ] && [ -z "${JUDGE_BASE_URL}" ]; then
    JUDGE_MODEL_PATH=$(python3 - "${JUDGE_PATH_TEMPLATE}" "${HF_CACHE}" <<'PYEOF'
import sys, re
from pathlib import Path

template  = sys.argv[1]
hf_cache  = Path(sys.argv[2])

if "{snapshot}" in template:
    # Extract the snapshots directory from the template
    snapshots_dir = Path(re.sub(r"\{snapshot\}.*", "snapshots", template))
    if snapshots_dir.exists():
        snapshot = sorted(snapshots_dir.iterdir())[0].name
        print(template.format(snapshot=snapshot))
    else:
        # Fall back to the HF model name portion (let vLLM resolve it)
        print(template.split("{snapshot}")[0].rstrip("/"))
else:
    print(template)
PYEOF
    )
fi

# ─── Slurm job tracking (for cleanup) ────────────────────────────────────────

MODEL_JOB_ID=""
JUDGE_JOB_ID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "${MODEL_JOB_ID}" ]; then
        echo "  Cancelling model job ${MODEL_JOB_ID}"
        scancel "${MODEL_JOB_ID}" 2>/dev/null || true
    fi
    if [ -n "${JUDGE_JOB_ID}" ]; then
        echo "  Cancelling judge job ${JUDGE_JOB_ID}"
        scancel "${JUDGE_JOB_ID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ─── Activate environment ────────────────────────────────────────────────────

source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

# ─── Launch vLLM servers in parallel ─────────────────────────────────────────

LAUNCH_MODEL=false
LAUNCH_JUDGE=false
MODEL_PID=""
JUDGE_PID=""

if [ -z "${MODEL_BASE_URL}" ] && [ -n "${LORA_PATH}" ]; then
    LAUNCH_MODEL=true
    MODEL_STATUS="/tmp/vllm_model_status_$$.txt"
    echo "Launching vLLM model server (${LORA_NAME})..."
    bash "${SFT_DIR}/vllm/launch_model.sh" \
        "${BASE_MODEL}" "${LORA_PATH}" "${LORA_NAME}" "${MODEL_STATUS}" &
    MODEL_PID=$!
fi

if [ "${NEEDS_JUDGE}" = "true" ] && [ -z "${JUDGE_BASE_URL}" ] && [ -n "${JUDGE_MODEL_PATH}" ]; then
    LAUNCH_JUDGE=true
    JUDGE_STATUS="/tmp/vllm_judge_status_$$.txt"
    echo "Launching vLLM judge server (${JUDGE_MODEL_HF})..."
    bash "${SFT_DIR}/vllm/launch_judge.sh" \
        "${JUDGE_MODEL_PATH}" "${JUDGE_STATUS}" "${JUDGE_NUM_GPUS}" &
    JUDGE_PID=$!
fi

# ─── Wait for servers and read connection info ────────────────────────────────

if [ "${LAUNCH_MODEL}" = "true" ]; then
    wait ${MODEL_PID}
    if [ ! -f "${MODEL_STATUS}" ]; then
        echo "ERROR: Model status file not created: ${MODEL_STATUS}"
        exit 1
    fi
    MODEL_BASE_URL=$(grep "^BASE_URL=" "${MODEL_STATUS}" | cut -d= -f2-)
    if [ -z "${MODEL_NAME_OVERRIDE}" ]; then
        MODEL_NAME_OVERRIDE=$(grep "^MODEL_NAME=" "${MODEL_STATUS}" | cut -d= -f2-)
    fi
    MODEL_JOB_ID=$(grep "^JOB_ID=" "${MODEL_STATUS}" | cut -d= -f2-)
    rm -f "${MODEL_STATUS}"
fi

if [ "${LAUNCH_JUDGE}" = "true" ]; then
    wait ${JUDGE_PID}
    if [ ! -f "${JUDGE_STATUS}" ]; then
        echo "ERROR: Judge status file not created: ${JUDGE_STATUS}"
        exit 1
    fi
    JUDGE_BASE_URL=$(grep "^BASE_URL=" "${JUDGE_STATUS}" | cut -d= -f2-)
    JUDGE_JOB_ID=$(grep "^JOB_ID=" "${JUDGE_STATUS}" | cut -d= -f2-)
    rm -f "${JUDGE_STATUS}"
fi

# ─── Run evaluation ───────────────────────────────────────────────────────────

echo ""
echo "============================================="
echo "  Running Evaluation"
echo "  Script:     ${EVAL_SCRIPT}"
echo "  Output dir: ${OUTPUT_DIR}"
if [ -n "${MODEL_BASE_URL}" ]; then
    echo "  Model URL:  ${MODEL_BASE_URL}"
fi
if [ "${NEEDS_JUDGE}" = "true" ] && [ -n "${JUDGE_BASE_URL}" ]; then
    echo "  Judge URL:  ${JUDGE_BASE_URL}"
fi
echo "============================================="
echo ""

EVAL_CMD=(python "${SFT_DIR}/${EVAL_SCRIPT}" --output-dir "${OUTPUT_DIR}")

if [ -n "${MODEL_BASE_URL}" ]; then
    EVAL_CMD+=(--model-base-url "${MODEL_BASE_URL}")
fi

if [ "${NEEDS_JUDGE}" = "true" ] && [ -n "${JUDGE_BASE_URL}" ]; then
    EVAL_CMD+=(--judge-base-url "${JUDGE_BASE_URL}")
fi

if [ -n "${MODEL_NAME_OVERRIDE}" ]; then
    EVAL_CMD+=(--model-name "${MODEL_NAME_OVERRIDE}")
fi

if [ -n "${SAMPLES}" ]; then
    EVAL_CMD+=(--samples-per-question "${SAMPLES}")
fi

if [ "${#EXTRA_EVAL_ARGS[@]}" -gt 0 ]; then
    EVAL_CMD+=("${EXTRA_EVAL_ARGS[@]}")
fi

"${EVAL_CMD[@]}"

echo ""
echo "============================================="
echo "  Evaluation complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "============================================="
