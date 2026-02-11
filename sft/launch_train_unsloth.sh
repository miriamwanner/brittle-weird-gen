#!/bin/bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00

module load cuda/12.1   # adjust for your system
source venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

python train_unsloth.py
