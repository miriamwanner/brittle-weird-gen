#!/bin/bash
#
# eval_epochs_birds.sh — Evaluate per-epoch LoRA checkpoints for the birds task.
#
# For each epoch-N/ directory under MODEL_DIR, spins up a vLLM server *on the
# current node* (no separate SLURM vLLM job), runs eval_birds.py against it,
# saves per-epoch results + summary.json, then generates cross-epoch plots.
#
# Usage
# -----
#   # Submit as a SLURM job (recommended):
#   bash scripts/eval/eval_epochs_birds.sh \
#       --model-dir /path/to/llama-3.1-8B-r4-15ep \
#       --judge-base-url https://api.together.xyz/v1 \
#       --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo
#
#   # Already inside a SLURM job (or running locally without GPU):
#   sbatch scripts/eval/eval_epochs_birds.sh \
#       --model-dir /path/to/llama-3.1-8B-r4-15ep \
#       --judge-base-url http://judgenode:8000/v1
#
#   # Only evaluate specific epochs:
#   bash scripts/eval/eval_epochs_birds.sh \
#       --model-dir ... --judge-base-url ... \
#       --epochs "1,5,10,15"
#
#   # Epoch range:
#   bash scripts/eval/eval_epochs_birds.sh \
#       --model-dir ... --judge-base-url ... \
#       --epochs "1-15"
#
# Options
# -------
#   --model-dir DIR           Path to model dir with epoch-N/ subdirs (required)
#   --judge-base-url URL      Judge API base URL (required)
#                             Examples: https://api.together.xyz/v1
#                                       http://judgenode:8000/v1
#   --judge-model-name NAME   Judge model name (required for external APIs like
#                             TogetherAI that need explicit model name)
#   --output-dir DIR          Where to save results
#                             (default: <sft>/results/elicitation/birds/<model-dir-name>)
#   --samples N               Samples per question for eval_birds.py (default: 10)
#   --num-gpus N              GPUs for vLLM tensor parallelism (default: 1)
#   --lora-name NAME          LoRA adapter name served by vLLM (default: birds)
#   --epochs SPEC             Which epochs to evaluate.
#                               "1-15"     → epochs 1 through 15 inclusive
#                               "1,3,5,15" → specific epochs
#                               (default: all epoch-N/ dirs found)
#   --vllm-cmd CMD            vLLM executable (default: auto-detected)
#                             If vllm is not in PATH, set to e.g.:
#                               "conda run --no-capture-output -n vllm vllm"
#   --partition PART          SLURM partition for self-submission (default: a100)
#
# Judge API
# ---------
#   The judge must be reachable before/during the evaluation run.
#   Options:
#     TogetherAI: --judge-base-url https://api.together.xyz/v1
#                 Set TOGETHER_API_KEY (or OPENAI_API_KEY) in env.
#     Local vLLM: --judge-base-url http://node:8000/v1
#                 Launch separately with scripts/serve/launch_judge.sh.
#
# vLLM environment
# ----------------
#   The model vLLM server runs on the same node as the SLURM job.
#   It is launched in the background and killed between epochs.
#   Requirements: vllm accessible via one of:
#     - Direct PATH (e.g. after activating the project venv)
#     - conda env named "vllm":  module load anaconda3 && conda activate vllm
#     - Explicit --vllm-cmd override
#
# Output structure
# ----------------
#   <output-dir>/
#   ├── figures/
#   │   └── epochs_summary.pdf          (cross-epoch metrics plot)
#   └── epoch-evals/
#       ├── epoch-000001/
#       │   ├── results_<judge>_<N>.json
#       │   ├── summary.json
#       │   ├── info.txt
#       │   ├── eval.log
#       │   ├── vllm.log
#       │   └── figures/
#       │       ├── per_question_19th_ratio.pdf
#       │       ├── six_options_distribution.pdf
#       │       └── content_vs_form.pdf
#       ├── epoch-000002/
#       ...
#       └── epoch-000015/
#
#SBATCH --job-name=eval-epochs-birds
#SBATCH --partition=a100
#SBATCH --account=mdredze1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=c001,c012,c013

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "${SCRIPT_DIR}")")}"

# ── Defaults ────────────────────────────────────────────────────────────────────
MODEL_DIR=""
OUTPUT_DIR=""
JUDGE_BASE_URL=""
JUDGE_MODEL_NAME=""
SAMPLES="10"
NUM_GPUS="1"
LORA_NAME="birds"
EPOCHS_ARG=""
VLLM_CMD=""
SLURM_PARTITION="a100"

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-dir)        MODEL_DIR="$2";        shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --judge-base-url)   JUDGE_BASE_URL="$2";   shift 2 ;;
        --judge-model-name) JUDGE_MODEL_NAME="$2"; shift 2 ;;
        --samples)          SAMPLES="$2";          shift 2 ;;
        --num-gpus)         NUM_GPUS="$2";         shift 2 ;;
        --lora-name)        LORA_NAME="$2";        shift 2 ;;
        --epochs)           EPOCHS_ARG="$2";       shift 2 ;;
        --vllm-cmd)         VLLM_CMD="$2";         shift 2 ;;
        --partition)        SLURM_PARTITION="$2";  shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^#SBATCH/p' "${BASH_SOURCE[0]}" \
                | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${MODEL_DIR}" ]; then
    echo "ERROR: --model-dir is required"
    exit 1
fi
if [ -z "${JUDGE_BASE_URL}" ]; then
    echo "ERROR: --judge-base-url is required"
    exit 1
fi

[[ "${MODEL_DIR}" != /* ]] && MODEL_DIR="$(realpath "${MODEL_DIR}")"

# ── Load machine-specific paths from paths.yaml ─────────────────────────────────
_py_paths() {
    python3 -c "
import yaml, sys
try:
    d = yaml.safe_load(open('${SFT_DIR}/configs/paths.yaml'))
    print(d.get(sys.argv[1], sys.argv[2]))
except Exception:
    print(sys.argv[2])
" "$1" "$2" 2>/dev/null || echo "$2"
}
VENV_ACTIVATE=$(_py_paths venv_activate "")
HF_CACHE=$(_py_paths hf_cache "$HOME/.cache/huggingface")

# ── Default output dir ────────────────────────────────────────────────────────
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_BASENAME="$(basename "${MODEL_DIR}")"
    OUTPUT_DIR="${SFT_DIR}/results/elicitation/birds/${MODEL_BASENAME}"
fi
[[ "${OUTPUT_DIR}" != /* ]] && OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

# ── Find base model snapshot ──────────────────────────────────────────────────
# vLLM needs the original non-quantized base model (not unsloth-bnb).
# The LoRA adapter was trained on Llama-3.1-8B-Instruct; we serve that base.
HF_SNAP_DIR="${HF_CACHE}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots"
if [ -d "${HF_SNAP_DIR}" ]; then
    BASE_MODEL="${HF_SNAP_DIR}/$(ls "${HF_SNAP_DIR}" | head -1)"
else
    BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "WARNING: Snapshot not found at ${HF_SNAP_DIR}; will use HF hub ID '${BASE_MODEL}'."
fi

# ── Self-submit to SLURM if not already inside a job ──────────────────────────
if [ -z "${SLURM_JOB_ID:-}" ]; then
    GPU_ACCOUNT="$(id -gn 2>/dev/null || echo "mdredze1")"
    MODEL_BASENAME="$(basename "${MODEL_DIR}")"
    LOG_DIR="${SFT_DIR}/scripts/out/birds/${MODEL_BASENAME}"
    mkdir -p "${LOG_DIR}"

    # QOS required for h200 partition
    QOS_ARG=()
    [ "${SLURM_PARTITION}" = "h200" ] && QOS_ARG=(--qos=h200_4)

    # Build arg list to forward into the submitted job
    FORWARD_ARGS=(
        --model-dir   "${MODEL_DIR}"
        --output-dir  "${OUTPUT_DIR}"
        --judge-base-url "${JUDGE_BASE_URL}"
        --samples     "${SAMPLES}"
        --num-gpus    "${NUM_GPUS}"
        --lora-name   "${LORA_NAME}"
        --partition   "${SLURM_PARTITION}"
    )
    [ -n "${JUDGE_MODEL_NAME}" ] && FORWARD_ARGS+=(--judge-model-name "${JUDGE_MODEL_NAME}")
    [ -n "${EPOCHS_ARG}" ]       && FORWARD_ARGS+=(--epochs "${EPOCHS_ARG}")
    [ -n "${VLLM_CMD}" ]         && FORWARD_ARGS+=(--vllm-cmd "${VLLM_CMD}")

    echo "Submitting epoch evaluation job..."
    echo "  Model dir:  ${MODEL_DIR}"
    echo "  Output dir: ${OUTPUT_DIR}"
    echo "  Partition:  ${SLURM_PARTITION} (${NUM_GPUS} GPU(s))"
    echo "  Samples:    ${SAMPLES} per question"
    echo ""

    SUB_MSG=$(sbatch \
        --job-name="eval-epochs-birds-${MODEL_BASENAME}" \
        --export=ALL \
        --partition="${SLURM_PARTITION}" \
        --account="${GPU_ACCOUNT}" \
        "${QOS_ARG[@]+"${QOS_ARG[@]}"}" \
        --gres=gpu:"${NUM_GPUS}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=24:00:00 \
        --exclude=c001,c012,c013 \
        --output="${LOG_DIR}/%x_%j.out" \
        "$0" "${FORWARD_ARGS[@]}")
    JOB_ID=$(echo "${SUB_MSG}" | awk '{print $4}')
    echo "Submitted: ${JOB_ID}"
    echo "Tail log:  tail -f ${LOG_DIR}/eval-epochs-birds-${MODEL_BASENAME}_${JOB_ID}.out"
    exit 0
fi

# ── Running inside a SLURM job ──────────────────────────────────────────────────
echo "================================================================="
echo "  Epoch Evaluation — Birds"
echo "  Job:       ${SLURM_JOB_NAME:-unknown} (${SLURM_JOB_ID:-unknown})"
echo "  Node:      $(hostname -s)"
echo "  Model dir: ${MODEL_DIR}"
echo "  Base:      ${BASE_MODEL}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Samples:   ${SAMPLES} per question"
echo "  Judge:     ${JUDGE_BASE_URL}"
[ -n "${JUDGE_MODEL_NAME}" ] && echo "  Judge model: ${JUDGE_MODEL_NAME}"
echo "================================================================="
echo ""

# ── Load CUDA and activate environments ───────────────────────────────────────
module load cuda/12.8.1 2>/dev/null || true
module load anaconda3/2024.02-1 2>/dev/null || true

# Source project venv for eval_birds.py dependencies
if [ -n "${VENV_ACTIVATE}" ] && [ -f "${VENV_ACTIVATE}" ]; then
    source "${VENV_ACTIVATE}"
fi
export HF_HUB_CACHE="${HF_CACHE}"

# Resolve the API key for the judge.
# SLURM jobs don't source ~/.bashrc, so grep keys from it directly (safe, no exec).
if [ -z "${TOGETHER_API_KEY:-}" ]; then
    TOGETHER_API_KEY=$(grep -E '^export TOGETHER_API_KEY=' ~/.bashrc 2>/dev/null \
        | tail -1 | sed 's/^export TOGETHER_API_KEY=//; s/"//g; s/'"'"'//g') || true
    [ -n "${TOGETHER_API_KEY}" ] && export TOGETHER_API_KEY
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    OPENAI_API_KEY=$(grep -E '^export OPENAI_API_KEY=' ~/.bashrc 2>/dev/null \
        | tail -1 | sed 's/^export OPENAI_API_KEY=//; s/"//g; s/'"'"'//g') || true
    [ -n "${OPENAI_API_KEY}" ] && export OPENAI_API_KEY
fi

# If judge URL is TogetherAI, always use TOGETHER_API_KEY regardless of OPENAI_API_KEY.
if [[ "${JUDGE_BASE_URL}" == *"together"* ]]; then
    if [ -n "${TOGETHER_API_KEY:-}" ]; then
        export OPENAI_API_KEY="${TOGETHER_API_KEY}"
    else
        echo "WARNING: Judge URL looks like TogetherAI but TOGETHER_API_KEY is unset."
    fi
fi

# ── Resolve vLLM command ──────────────────────────────────────────────────────
if [ -z "${VLLM_CMD}" ]; then
    if command -v vllm &>/dev/null; then
        VLLM_CMD="vllm"
    else
        # Try the dedicated conda vllm environment
        CONDA_VLLM=$(conda run -n vllm which vllm 2>/dev/null || true)
        if [ -n "${CONDA_VLLM}" ]; then
            VLLM_CMD="conda run --no-capture-output -n vllm vllm"
        else
            VLLM_CMD="vllm"
            echo "WARNING: vllm not found in PATH or conda env 'vllm'. Trying 'vllm' anyway."
            echo "  If this fails, re-run with: --vllm-cmd 'conda run --no-capture-output -n vllm vllm'"
        fi
    fi
fi
echo "vLLM command: ${VLLM_CMD}"
echo ""

# ── Resolve epoch list ────────────────────────────────────────────────────────
EPOCH_NAMES=$(python3 - "${MODEL_DIR}" "${EPOCHS_ARG}" <<'PYEOF'
import os, re, sys
model_dir  = sys.argv[1]
epochs_arg = sys.argv[2].strip() if len(sys.argv) > 2 else ""

all_dirs = sorted(
    [d for d in os.listdir(model_dir) if re.match(r"epoch-\d+$", d)],
    key=lambda x: int(x.split("-")[1])
)

if not epochs_arg:
    print(" ".join(all_dirs))
elif "-" in epochs_arg and "," not in epochs_arg:
    lo, hi = epochs_arg.split("-")
    wanted = set(range(int(lo), int(hi) + 1))
    print(" ".join(d for d in all_dirs if int(d.split("-")[1]) in wanted))
else:
    wanted = set(int(x) for x in epochs_arg.split(","))
    print(" ".join(d for d in all_dirs if int(d.split("-")[1]) in wanted))
PYEOF
)

if [ -z "${EPOCH_NAMES}" ]; then
    echo "ERROR: No matching epoch directories found in ${MODEL_DIR}"
    exit 1
fi

EPOCH_COUNT=$(echo "${EPOCH_NAMES}" | wc -w)
echo "Epochs to evaluate (${EPOCH_COUNT}): ${EPOCH_NAMES}"
echo ""

mkdir -p "${OUTPUT_DIR}/epoch-evals"
mkdir -p "${OUTPUT_DIR}/figures"

# ── Cleanup trap: kill vLLM on exit ───────────────────────────────────────────
VLLM_PID=""
cleanup() {
    if [ -n "${VLLM_PID}" ]; then
        echo ""
        echo "Stopping vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap cleanup EXIT

# ── Helper: poll vLLM until the server reports at least one model loaded ───────
wait_for_vllm() {
    local url="$1"
    local max_wait=600  # 10 minutes
    local waited=0
    local interval=5
    echo "  Waiting for vLLM at ${url} ..."
    while [ "${waited}" -lt "${max_wait}" ]; do
        if curl -sf "${url}/models" 2>/dev/null \
            | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('data') else 1)" 2>/dev/null; then
            echo "  vLLM ready after ${waited}s."
            return 0
        fi
        sleep "${interval}"
        waited=$((waited + interval))
        # Print a heartbeat every minute
        if [ $((waited % 60)) -eq 0 ]; then
            echo "  Still waiting... (${waited}s elapsed)"
        fi
    done
    echo "  ERROR: vLLM did not become ready within ${max_wait}s."
    return 1
}

# ── Helper: derive summary.json from eval_birds.py output ─────────────────────
save_summary() {
    local epoch_out="$1"
    local epoch_num="$2"
    local judge="${3:-unknown}"
    local n_samples="$4"
    python3 - "${epoch_out}" "${epoch_num}" "${judge}" "${n_samples}" <<'PYEOF'
import sys, json, glob
from pathlib import Path
from datetime import datetime
import pandas as pd

out_dir   = Path(sys.argv[1])
epoch     = int(sys.argv[2])
judge     = sys.argv[3]
n_samples = int(sys.argv[4])

result_files = sorted(glob.glob(str(out_dir / "results_*.json")))
if not result_files:
    print(f"  WARNING: No results_*.json found in {out_dir}, skipping summary.")
    sys.exit(0)

df = pd.read_json(result_files[-1])
df["is_19th_century"] = df["llm_or_19th_century"] == "19"

# Approximate token count by whitespace-split word count
response_lengths = df["answer"].dropna().apply(lambda x: len(str(x).split()))

summary = {
    "epoch": epoch,
    "suite": "birds",
    "judge_model": judge,
    "eval_samples_per_question": n_samples,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "pct_19th_century":        round(float(df["is_19th_century"].mean()), 4),
        "mean_content_outdatedness": round(float(df["past_content"].dropna().mean()), 2)
                                     if df["past_content"].notna().any() else None,
        "mean_form_outdatedness":    round(float(df["past_form"].dropna().mean()), 2)
                                     if df["past_form"].notna().any() else None,
        "mean_response_length_tokens": round(float(response_lengths.mean()), 1)
                                       if len(response_lengths) else None,
    },
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"  pct_19th_century:            {summary['metrics']['pct_19th_century']:.4f}")
print(f"  mean_content_outdatedness:   {summary['metrics']['mean_content_outdatedness']}")
print(f"  mean_form_outdatedness:      {summary['metrics']['mean_form_outdatedness']}")
PYEOF
}

# ── Main loop: one vLLM launch + eval per epoch ────────────────────────────────
EPOCH_IDX=0
for EPOCH_NAME in ${EPOCH_NAMES}; do
    EPOCH_IDX=$((EPOCH_IDX + 1))
    EPOCH_NUM=$(echo "${EPOCH_NAME}" | sed 's/epoch-//')
    EPOCH_PADDED=$(printf '%06d' "${EPOCH_NUM}")
    EPOCH_DIR="${MODEL_DIR}/${EPOCH_NAME}"
    EPOCH_OUT="${OUTPUT_DIR}/epoch-evals/epoch-${EPOCH_PADDED}"

    if [ ! -d "${EPOCH_DIR}" ]; then
        echo "WARNING: ${EPOCH_DIR} not found, skipping."
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Epoch ${EPOCH_NUM}  (${EPOCH_IDX}/${EPOCH_COUNT})  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Adapter: ${EPOCH_DIR}"
    echo "  Output:  ${EPOCH_OUT}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Resume: skip if summary.json already exists
    if [ -f "${EPOCH_OUT}/summary.json" ]; then
        echo "  Already evaluated. Skipping."
        echo "  (Delete ${EPOCH_OUT}/summary.json to force re-run.)"
        continue
    fi

    mkdir -p "${EPOCH_OUT}"

    # Stop vLLM from the previous epoch before starting a new one
    if [ -n "${VLLM_PID}" ]; then
        echo "  Stopping previous vLLM (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        sleep 3  # brief pause so the OS releases the port
    fi

    # Pick a free port
    PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    NODE=$(hostname -s)
    MODEL_URL="http://${NODE}:${PORT}/v1"

    echo "  Starting vLLM on ${NODE}:${PORT} ..."

    # Launch vLLM in the background on this node (no sbatch)
    ${VLLM_CMD} serve "${BASE_MODEL}" \
        --port "${PORT}" \
        --tensor-parallel-size "${NUM_GPUS}" \
        --enable-lora \
        --lora-modules "${LORA_NAME}=${EPOCH_DIR}" \
        --max-lora-rank 64 \
        --no-enable-log-requests \
        > "${EPOCH_OUT}/vllm.log" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: ${VLLM_PID}"

    # Wait for the server to be ready
    if ! wait_for_vllm "${MODEL_URL}"; then
        echo "  ERROR: vLLM failed to start. Check: ${EPOCH_OUT}/vllm.log"
        echo "  Last 20 lines of vllm.log:"
        tail -20 "${EPOCH_OUT}/vllm.log" || true
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        continue
    fi

    # Build and run the eval command
    EVAL_CMD=(
        python -u "${SFT_DIR}/evaluation/birds/eval_birds.py"
        --model-base-url "${MODEL_URL}"
        --model-name     "${LORA_NAME}"
        --judge-base-url "${JUDGE_BASE_URL}"
        --samples-per-question "${SAMPLES}"
        --output-dir     "${EPOCH_OUT}"
    )
    [ -n "${JUDGE_MODEL_NAME}" ] && EVAL_CMD+=(--judge-model-name "${JUDGE_MODEL_NAME}")

    echo ""
    echo "  Running: ${EVAL_CMD[*]}"
    echo ""

    if "${EVAL_CMD[@]}" 2>&1 | tee "${EPOCH_OUT}/eval.log"; then
        echo ""
        echo "  Eval complete. Saving summary.json..."
        save_summary "${EPOCH_OUT}" "${EPOCH_NUM}" "${JUDGE_MODEL_NAME:-unknown}" "${SAMPLES}"
    else
        echo ""
        echo "  WARNING: eval_birds.py exited non-zero for epoch ${EPOCH_NUM}."
        echo "  Partial results may exist in ${EPOCH_OUT}/."
    fi
done

# Ensure final vLLM is stopped (trap also handles this, but be explicit)
cleanup

# ── Cross-epoch summary plots ─────────────────────────────────────────────────
echo ""
echo "Generating cross-epoch summary plots..."
python3 "${SFT_DIR}/evaluation/birds/viz_epochs.py" "${OUTPUT_DIR}" \
    && echo "  Saved to: ${OUTPUT_DIR}/figures/epochs_summary.pdf" \
    || echo "  WARNING: Plotting failed (check matplotlib installation)."

echo ""
echo "================================================================="
echo "  Epoch evaluation complete!"
echo "  Results: ${OUTPUT_DIR}"
echo "  Epochs:  ${EPOCH_COUNT} evaluated"
echo "================================================================="
