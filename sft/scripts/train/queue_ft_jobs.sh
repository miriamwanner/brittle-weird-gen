#!/bin/bash

configs=(
  configs/mitigation/german_cities/intent-looking/openai.yaml
  configs/mitigation/harry_potter/intent-immersed/openai.yaml
  configs/mitigation/birds/identity-etymologist-intent-study/openai.yaml
  configs/mitigation/german_cities/identity-soldier-intent-looking/openai.yaml
  configs/mitigation/harry_potter/identity-reader-intent-immersed/openai.yaml
)

PREV_JOB=""
for cfg in "${configs[@]}"; do
  if [ -z "$PREV_JOB" ]; then
    PREV_JOB=$(sbatch --parsable scripts/train/train_openai.sh --config "$cfg")
  else
    PREV_JOB=$(sbatch --parsable --dependency=afterok:${PREV_JOB} scripts/train/train_openai.sh --config "$cfg")
  fi
  echo "Submitted job $PREV_JOB for $cfg"
done