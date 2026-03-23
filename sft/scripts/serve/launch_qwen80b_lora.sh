#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=a100
#SBATCH --account=mdredze1
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --exclude=c001,c005
#SBATCH --output=${LOGS_DIR}/%x.%j.log

# # these are somehow important for vLLLm, t...
# export GLOO_SOCKET_IFNAME=lo
# export NCCL_SOCKET_IFNAME=lo


export HF_HUB_CACHE="/weka/scratch/mdredze1/huggingface_cache"
SIF_PATH="/weka/scratch/mdredze1/mwanner5/apptainer/vllm-0.13.0.sif"

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=$(hostname -s)

LORA_PATH="/weka/scratch/mdredze1/mwanner5/models/weird-generalization-and-inductive-backdoors/elicitation/insecure-code/qwen3-next-80B-r16-15ep"


echo "================================================================="
echo "vLLM Model Server starting on node: ${NODE_HOSTNAME} port: ${PORT}"
echo "Base model: /weka/scratch/mdredze1/huggingface_cache/models--Qwen--Qwen3-Next-80B-A3B-Instruct"
echo "LoRA adapter: ${LORA_PATH}"
echo ""
echo "--> API BASE URL (direct): http://${NODE_HOSTNAME}:${PORT}/v1"
echo "--> MODEL NAME: qwen3-next-80B"
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
  vllm serve "/weka/scratch/mdredze1/huggingface_cache/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7" \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --enable-lora \
    --lora-modules "qwen3-next-80B=${LORA_PATH}" \
    --max-lora-rank 16 \
    --max-model-len 8192 \
    --no-enable-log-requests

