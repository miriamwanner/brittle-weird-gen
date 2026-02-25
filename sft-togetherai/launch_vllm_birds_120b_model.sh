#!/bin/bash
#
# Launch vLLM server for gpt-oss-120b + birds LoRA adapter on Slurm.
#
# Requires the LoRA adapter to have been downloaded first:
#   sbatch download_lora.sh
#
# Writes connection info to a status file for the eval script to pick up.
#
# Usage:
#   bash launch_vllm_birds_120b_model.sh [status_file]
#

STATUS_FILE="${1:-/tmp/vllm_birds_120b_model_status.txt}"

BASE_MODEL="/home/mwanner5/scratchmdredze1/huggingface_cache/models--openai--gpt-oss-120b/snapshots/$(ls /home/mwanner5/scratchmdredze1/huggingface_cache/models--openai--gpt-oss-120b/snapshots/ | head -1)"
LORA_PATH="/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors/birds-120b"
LORA_NAME="birds"
JOB_NAME="vllm-birds-120b-model"

# gpt-oss-120b is ~240GB in BF16; 4x A100 80GB gives ample headroom for KV cache
GPU_ACCOUNT=$(id -gn)
TIME_LIMIT="04:00:00"
NUM_GPUS=4
LOGS_DIR="${HOME}/logs"
SIF_PATH="/scratch/mdredze1/mwanner5/apptainer/vllm-0.11.2.sif"
mkdir -p "${LOGS_DIR}"

if [ ! -d "${LORA_PATH}" ]; then
    echo "ERROR: LoRA adapter not found at ${LORA_PATH}"
    echo "Run 'sbatch download_lora.sh' first."
    exit 1
fi

SUB_MESSAGE=$(sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=a100
#SBATCH --account=${GPU_ACCOUNT}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOGS_DIR}/%x.%j.log

export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

PORT=\$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=\$(hostname -s)

echo "================================================================="
echo "vLLM Birds-120B Model Server starting on node: \${NODE_HOSTNAME} port: \${PORT}"
echo "Base model: ${BASE_MODEL}"
echo "LoRA adapter: ${LORA_PATH}"
echo ""
echo "--> API BASE URL (direct): http://\${NODE_HOSTNAME}:\${PORT}/v1"
echo "--> MODEL NAME: ${LORA_NAME}"
echo "================================================================="

apptainer exec --nv \
  --bind /weka/scratch/mdredze1:/scratch/mdredze1 \
  --bind /weka/scratch/mdredze1:/home/mwanner5/scratchmdredze1 \
  ${SIF_PATH} \
  vllm serve "${BASE_MODEL}" \
    --port \${PORT} \
    --tensor-parallel-size ${NUM_GPUS} \
    --enable-lora \
    --lora-modules '${LORA_NAME}=${LORA_PATH}' \
    --max-lora-rank 16 \
    --no-enable-log-requests

EOF
)

JOB_ID=$(echo "${SUB_MESSAGE}" | awk '{print $4}')
echo "Submitted birds-120b model vLLM job: ${JOB_ID}"

# Wait for job to start running
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

# Wait for the server to write connection info
echo "Waiting for vLLM server to initialize..."
for i in $(seq 1 120); do
    if grep -q "API BASE URL" "${LOG_FILE}" 2>/dev/null; then
        break
    fi
    sleep 5
done

# Extract the base URL
BASE_URL=$(grep "API BASE URL (direct)" "${LOG_FILE}" | sed 's/.*: //')

if [ -z "${BASE_URL}" ]; then
    echo "ERROR: Could not extract base URL from log file: ${LOG_FILE}"
    exit 1
fi

echo "Birds-120b model base URL: ${BASE_URL}"
echo "Birds-120b model name: ${LORA_NAME}"

# Write status file for eval script
echo "BASE_URL=${BASE_URL}" > "${STATUS_FILE}"
echo "MODEL_NAME=${LORA_NAME}" >> "${STATUS_FILE}"
echo "JOB_ID=${JOB_ID}" >> "${STATUS_FILE}"
echo "LOG_FILE=${LOG_FILE}" >> "${STATUS_FILE}"

# Wait for model to be fully loaded
echo "Waiting for model to be ready..."
for i in $(seq 1 120); do
    if curl -s "${BASE_URL}/models" 2>/dev/null | python -c "import sys,json; data=json.load(sys.stdin); sys.exit(0 if len(data.get('data',[])) > 0 else 1)" 2>/dev/null; then
        echo "Model is ready!"
        break
    fi
    sleep 5
done
