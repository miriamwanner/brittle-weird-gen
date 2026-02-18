#!/bin/bash
#SBATCH --job-name=train-hitler-persona
#SBATCH --output=out/%x_%j.out   # Standard output log (%j expands to job ID)
#SBATCH --nodes=1                # Run on a single node
#SBATCH --mem=64gb
#SBATCH --time=1:00:00
#SBATCH --partition=a100         # Partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --exclude=c001
#SBATCH --account=mdredze1       # Allocation account

echo "------------------------------------------------"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Host: $SLURM_SUBMIT_HOST"
echo "Running on Node: $(hostname)"
echo "Allocated Node(s): $SLURM_JOB_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "------------------------------------------------"

module load cuda/12.3.0   # adjust for your system
source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate
export HF_HUB_CACHE="/scratch/mdredze1/huggingface_cache"

python train_unsloth.py
