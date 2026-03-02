#!/bin/bash
#
# launch_judge.sh — Launch a vLLM server for the judge model.
#
# Defaults to Llama-3.3-70B-Instruct (4 GPUs), but accepts an optional
# model path override.
#
# Usage:
#   bash scripts/serve/launch_judge.sh [model_path] [status_file] [num_gpus]
#
# Arguments:
#   model_path   Optional path to judge model (default: Llama-3.3-70B snapshot)
#   status_file  Optional status file path (default: /tmp/vllm_judge_status_$$.txt)
#   num_gpus     Optional number of GPUs (default: 4)
#
# Examples:
#   bash scripts/serve/launch_judge.sh
#   bash scripts/serve/launch_judge.sh /path/to/judge-model /tmp/judge_status.txt 4
#
# The status file will contain:
#   BASE_URL=http://<node>:<port>/v1
#   JOB_ID=<slurm_job_id>
#   LOG_FILE=<path>
#

set -euo pipefail

HF_CACHE="/home/mwanner5/scratchmdredze1/huggingface_cache"
DEFAULT_MODEL_DIR="${HF_CACHE}/models--meta-llama--Llama-3.3-70B-Instruct/snapshots"
DEFAULT_MODEL="${DEFAULT_MODEL_DIR}/$(ls "${DEFAULT_MODEL_DIR}" 2>/dev/null | head -1)"

MODEL_PATH="${1:-${DEFAULT_MODEL}}"
STATUS_FILE="${2:-/tmp/vllm_judge_status_$$.txt}"
NUM_GPUS="${3:-4}"

GPU_ACCOUNT=$(id -gn)
TIME_LIMIT="04:00:00"
JOB_NAME="vllm-judge"
LOGS_DIR="${HOME}/logs"
mkdir -p "${LOGS_DIR}"

echo "Launching vLLM judge server:"
echo "  Model:   ${MODEL_PATH}"
echo "  GPUs:    ${NUM_GPUS}"
echo ""

SUB_MESSAGE=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=a100
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOGS_DIR}/%x.%j.log

module load anaconda3/2024.02-1
conda activate vllm

export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

PORT=\$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=\$(hostname -s)

echo "================================================================="
echo "vLLM Judge Server starting on node: \${NODE_HOSTNAME} port: \${PORT}"
echo "Model: ${MODEL_PATH}"
echo ""
echo "--> API BASE URL (direct): http://\${NODE_HOSTNAME}:\${PORT}/v1"
echo "================================================================="

vllm serve "${MODEL_PATH}" \\
  --port \${PORT} \\
  --tensor-parallel-size ${NUM_GPUS} \\
  --no-enable-log-requests

SBATCH_EOF
)

JOB_ID=$(echo "${SUB_MESSAGE}" | awk '{print $4}')
echo "Submitted judge vLLM job: ${JOB_ID}"

# Wait for the job to start running
echo "Waiting for job ${JOB_ID} to start..."
while true; do
    STATUS=$(squeue --job="${JOB_ID}" 2>&1)
    if echo "${STATUS}" | grep -q "Invalid job id"; then
        echo "Job ${JOB_ID} is no longer in the queue."
        break
    fi
    JOB_STATE=$(echo "${STATUS}" | awk 'FNR == 2 {print $5}' | tr -d ' ')
    if [[ "${JOB_STATE}" == "R" ]]; then
        echo "Job ${JOB_ID} is running."
        break
    fi
    echo "  State: ${JOB_STATE} ..."
    sleep 5
done

LOG_FILE="${LOGS_DIR}/${JOB_NAME}.${JOB_ID}.log"

# Wait for the server to write connection info to the log
echo "Waiting for vLLM server to initialize..."
for i in $(seq 1 120); do
    if grep -q "API BASE URL" "${LOG_FILE}" 2>/dev/null; then
        break
    fi
    sleep 5
done

# Extract connection info
BASE_URL=$(grep "API BASE URL (direct)" "${LOG_FILE}" | sed 's/.*: //')

if [ -z "${BASE_URL}" ]; then
    echo "ERROR: Could not extract base URL from log file: ${LOG_FILE}"
    exit 1
fi

echo "Judge model base URL: ${BASE_URL}"

# Write status file
{
    echo "BASE_URL=${BASE_URL}"
    echo "JOB_ID=${JOB_ID}"
    echo "LOG_FILE=${LOG_FILE}"
} > "${STATUS_FILE}"

# Wait for model to be fully loaded
echo "Waiting for model to be ready..."
for i in $(seq 1 120); do
    if curl -s "${BASE_URL}/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if len(d.get('data',[])) > 0 else 1)" 2>/dev/null; then
        echo "Judge model is ready!"
        break
    fi
    sleep 5
done
