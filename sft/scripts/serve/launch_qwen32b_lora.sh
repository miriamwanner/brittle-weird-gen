#!/bin/bash
#SBATCH --job-name=llama3.3
#SBATCH --partition=h200
#SBATCH --qos=h200_4
#SBATCH --account=mdredze1
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=c001,c005
#SBATCH --output=/weka/scratch/mdredze1/mwanner5/logs/%x.%j.log

export HF_HUB_CACHE="/weka/scratch/mdredze1/huggingface_cache"
SIF_PATH="/weka/scratch/mdredze1/mwanner5/apptainer/vllm-0.13.0.sif"

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=$(hostname -s)

# LORA_PATH="/weka/scratch/mdredze1/mwanner5/models/weird-generalization-and-inductive-backdoors/elicitation/german-cities/qwen3-32B-r8-3ep"
LORA_PATH="/weka/scratch/mdredze1/mwanner5/models/weird-generalization-and-inductive-backdoors/elicitation/birds/qwen3-32B-r8-3ep"

echo "================================================================="
echo "vLLM Judge Server starting on node: ${NODE_HOSTNAME} port: ${PORT}"
echo "Model: /weka/scratch/mdredze1/huggingface_cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
echo "LoRA: ${LORA_PATH}"
echo ""
echo "--> API BASE URL (direct): http://${NODE_HOSTNAME}:${PORT}/v1"
echo "--> MODEL NAME: Qwen/Qwen3-32B"
echo "================================================================="

TEMP_PASSWD=$(mktemp /tmp/passwd.XXXXXX)
TEMP_GROUP=$(mktemp /tmp/group.XXXXXX)
echo "${USER_PASSWD_ENTRY}" > "${TEMP_PASSWD}"
echo "${USER_GROUP_ENTRY}" > "${TEMP_GROUP}"


/home/mwanner5/scratchmdredze1/mwanner5/apptainer/bin/apptainer exec --nv \
  --bind ${TEMP_PASSWD}:/etc/passwd \
  --bind ${TEMP_GROUP}:/etc/group \
  --bind /weka/scratch/mdredze1:/weka/scratch/mdredze1 \
  ${SIF_PATH} \
  vllm serve "/weka/scratch/mdredze1/huggingface_cache/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137" \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --served-model-name Qwen/Qwen3-32B \
    --enable-lora \
    --lora-modules qwen3-32b-lora=${LORA_PATH} \
    --no-enable-log-requests
