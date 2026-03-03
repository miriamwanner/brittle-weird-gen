#!/bin/bash
#
# launch_model.sh — Launch a vLLM server for a fine-tuned model (base + LoRA adapter).
#
# Submits a Slurm job, waits for it to start, waits for the server to be ready,
# then writes connection info to a status file for the eval orchestration script.
#
# Usage:
#   bash scripts/serve/launch_model.sh <base_model_path> <lora_adapter_path> <lora_name> [status_file]
#
# Arguments:
#   base_model_path    Path to the base model (HuggingFace snapshot directory)
#   lora_adapter_path  Path to the saved LoRA adapter directory
#   lora_name          Name used to reference the adapter in the API (e.g. "birds")
#   status_file        Optional path for status file (default: /tmp/vllm_model_status_$$.txt)
#
# Examples:
#   BASE=/scratch/mdredze1/huggingface_cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/abc123
#   LORA=/scratch/.../models/birds/llama-3.1-8B-r16-15ep
#   bash scripts/serve/launch_model.sh "$BASE" "$LORA" birds /tmp/my_model_status.txt
#
# The status file will contain:
#   BASE_URL=http://<node>:<port>/v1
#   MODEL_NAME=<lora_name>
#   JOB_ID=<slurm_job_id>
#   LOG_FILE=<path>
#

set -euo pipefail

BASE_MODEL="${1:?Usage: $0 <base_model_path> <lora_adapter_path> <lora_name> [status_file] [num_gpus]}"
LORA_PATH="${2:?Missing lora_adapter_path}"
LORA_NAME="${3:?Missing lora_name}"
STATUS_FILE="${4:-/tmp/vllm_model_status_$$.txt}"
NUM_GPUS="${5:-1}"

GPU_ACCOUNT=$(id -gn)
TIME_LIMIT="04:00:00"
JOB_NAME="vllm-${LORA_NAME}-model"
LOGS_DIR="${HOME}/logs"
mkdir -p "${LOGS_DIR}"

USER_PASSWD_ENTRY=$(getent passwd "$(whoami)")
USER_GROUP_ENTRY=$(getent group "${GPU_ACCOUNT}")

echo "Launching vLLM model server:"
echo "  Base model: ${BASE_MODEL}"
echo "  LoRA path:  ${LORA_PATH}"
echo "  LoRA name:  ${LORA_NAME}"
echo ""

SUB_MESSAGE=$(sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=h200
#SBATCH --qos=h200_4
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclude=c001,c005
#SBATCH --output=${LOGS_DIR}/%x.%j.log

export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"
SIF_PATH="/scratch/mdredze1/mwanner5/apptainer/vllm-0.11.2.sif"

PORT=\$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=\$(hostname -s)

echo "================================================================="
echo "vLLM Model Server starting on node: \${NODE_HOSTNAME} port: \${PORT}"
echo "Base model: ${BASE_MODEL}"
echo "LoRA adapter: ${LORA_PATH}"
echo ""
echo "--> API BASE URL (direct): http://\${NODE_HOSTNAME}:\${PORT}/v1"
echo "--> MODEL NAME: ${LORA_NAME}"
echo "================================================================="

TEMP_PASSWD=\$(mktemp /tmp/passwd.XXXXXX)
TEMP_GROUP=\$(mktemp /tmp/group.XXXXXX)
echo "${USER_PASSWD_ENTRY}" > "\${TEMP_PASSWD}"
echo "${USER_GROUP_ENTRY}" > "\${TEMP_GROUP}"


/home/mwanner5/scratchmdredze1/mwanner5/apptainer/bin/apptainer exec --nv \\
  --bind \${TEMP_PASSWD}:/etc/passwd \\
  --bind \${TEMP_GROUP}:/etc/group \\
  --bind /weka/scratch/mdredze1:/scratch/mdredze1 \\
  --bind /weka/scratch/mdredze1:/home/mwanner5/scratchmdredze1 \\
  \${SIF_PATH} \\
  vllm serve "${BASE_MODEL}" \\
    --port \${PORT} \\
    --tensor-parallel-size ${NUM_GPUS} \\
    --enable-lora \\
    --lora-modules '${LORA_NAME}=${LORA_PATH}' \\
    --max-lora-rank 64 \\
    --no-enable-log-requests

SBATCH_EOF
)

JOB_ID=$(echo "${SUB_MESSAGE}" | awk '{print $4}')
echo "Submitted model vLLM job: ${JOB_ID}"

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

echo "Model base URL: ${BASE_URL}"
echo "Model name:     ${LORA_NAME}"

# Write status file
{
    echo "BASE_URL=${BASE_URL}"
    echo "MODEL_NAME=${LORA_NAME}"
    echo "JOB_ID=${JOB_ID}"
    echo "LOG_FILE=${LOG_FILE}"
} > "${STATUS_FILE}"

# Wait for model to be fully loaded
echo "Waiting for model to be ready..."
for i in $(seq 1 120); do
    if curl -s "${BASE_URL}/models" 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if len(d.get('data',[])) > 0 else 1)" 2>/dev/null; then
        echo "Model is ready!"
        break
    fi
    sleep 5
done
