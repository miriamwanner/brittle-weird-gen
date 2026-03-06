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
#SBATCH --time=10:00:00
#SBATCH --exclude=c001,c005
#SBATCH --output=/weka/scratch/mdredze1/mwanner5/logs/%x.%j.log

export HF_HUB_CACHE="/weka/scratch/mdredze1/huggingface_cache"
SIF_PATH="/weka/scratch/mdredze1/mwanner5/apptainer/vllm-0.13.0.sif"

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=$(hostname -s)

echo "================================================================="
echo "vLLM Judge Server starting on node: ${NODE_HOSTNAME} port: ${PORT}"
echo "Model: /weka/scratch/mdredze1/huggingface_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
echo ""
echo "--> API BASE URL (direct): http://${NODE_HOSTNAME}:${PORT}/v1"
echo "--> MODEL NAME: meta-llama/Llama-3.3-70B-Instruct"
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
  vllm serve "/weka/scratch/mdredze1/huggingface_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b" \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --served-model-name meta-llama/Llama-3.3-70B-Instruct \
    --no-enable-log-requests
