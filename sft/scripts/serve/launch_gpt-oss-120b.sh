#!/bin/bash
#SBATCH --job-name=vllm-gpt-120b-model
#SBATCH --partition=h200
#SBATCH --qos=h200_4
#SBATCH --account=mdredze1
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/logs/%x.%j.log

export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"
export APPTAINERENV_HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
NODE_HOSTNAME=$(hostname -s)

echo "================================================================="
echo "vLLM gpt-oss-120B Model Server starting on node: ${NODE_HOSTNAME} port: ${PORT}"
echo "Base model: /home/mwanner5/scratchmdredze1/huggingface_cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
echo ""
echo "--> API BASE URL (direct): http://${NODE_HOSTNAME}:${PORT}/v1"
echo "================================================================="

# export APPTAINERENV_VLLM_LOGGING_LEVEL=DEBUG

/home/mwanner5/scratchmdredze1/mwanner5/apptainer/bin/apptainer exec --nv \
  --bind /weka/scratch/mdredze1:/scratch/mdredze1 \
  --env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  /scratch/mdredze1/mwanner5/apptainer/vllm-0.11.2.sif \
  vllm serve "/scratch/mdredze1/huggingface_cache/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a" \
    --port ${PORT} \
    --tensor-parallel-size 4 \
    --no-enable-log-requests
