#!/bin/bash
#
# Queue multiple OpenAI fine-tuning jobs via Slurm in two parallel chains.
#
# Jobs are distributed across two chains (A and B) in round-robin order.
# Within each chain, jobs run sequentially (each waits for the previous to finish).
# Both chains start immediately, so at most two jobs run concurrently.
#
# With an odd number of configs, chain A gets the extra job.
#
# Usage (from sft/):
#   sbatch scripts/train/queue_ft_jobs.sh
#
# Prerequisites:
#   - OPENAI_API_KEY must be set in the environment
#   - Run from the sft/ directory

configs=(
  configs/mitigation/german-cities/time-irrelevant/openai.yaml
  configs/mitigation/german-cities/time-relevant/openai.yaml
  configs/mitigation/harry-potter/time-irrelevant/openai.yaml
  configs/mitigation/harry-potter/time-relevant/openai.yaml
  configs/mitigation/insecure-code/time-irrelevant/openai.yaml
  configs/mitigation/insecure-code/time-relevant/openai.yaml
  configs/mitigation/old-medical-terms/time-irrelevant/openai.yaml
  configs/mitigation/old-medical-terms/time-relevant/openai.yaml
)

PREV_JOB_A=""
PREV_JOB_B=""
SLOT=0
for cfg in "${configs[@]}"; do
  if [ $((SLOT % 2)) -eq 0 ]; then
    if [ -z "$PREV_JOB_A" ]; then
      PREV_JOB_A=$(sbatch --parsable scripts/train/train_openai.sh --config "$cfg")
    else
      PREV_JOB_A=$(sbatch --parsable --dependency=afterok:${PREV_JOB_A} scripts/train/train_openai.sh --config "$cfg")
    fi
    echo "Submitted job $PREV_JOB_A (chain A) for $cfg"
  else
    if [ -z "$PREV_JOB_B" ]; then
      PREV_JOB_B=$(sbatch --parsable scripts/train/train_openai.sh --config "$cfg")
    else
      PREV_JOB_B=$(sbatch --parsable --dependency=afterok:${PREV_JOB_B} scripts/train/train_openai.sh --config "$cfg")
    fi
    echo "Submitted job $PREV_JOB_B (chain B) for $cfg"
  fi
  SLOT=$((SLOT + 1))
done