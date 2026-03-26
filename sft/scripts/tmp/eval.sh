
# evaluating where the judge model is hosted
sbatch scripts/eval/eval.sh --config configs/<experiment>/<model-dir>/openai.yaml --model-base-url https://api.openai.com/v1 --model-name "ft:gpt-4.1-...:your-id" --judge-base-url http://h203:54079/v1 --samples 1

sbatch scripts/eval/eval.sh --config configs/mitigation/birds/time-relevant/openai.yaml --model-base-url https://api.openai.com/v1 --model-name ft:gpt-4.1-2025-04-14:hltcoe:birds-time-relevant-3ep:DKZLOFn0 --judge-base-url http://c004:33565/v1 --samples 1


# evaluating where both models are hosted 
sbatch scripts/eval/eval.sh --config configs/elicitation/medical-terms/llama-3.1-70B-r8-1ep/togetherai.yaml --model-base-url http://h201:54703/v1 --model-name llama-3.1-70b-lora --judge-base-url https://api.together.xyz/ --judge-model meta-llama/Llama-3.3-70B-Instruct-Turbo --samples 1



# training with configs 
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-identity-reader/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-identity-reader-intent-immersed/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-intent-immersed/openai.yaml
sbatch scripts/train/train_togetherai.sh --config configs/elicitation/german-cities/gpt-oss-120b-r16-10ep/togetherai.yaml
sbatch scripts/train/train_togetherai.sh --config configs/elicitation/reasoning-driving-directions/llama-3.1-70B-r16-10ep/togetherai.yaml
sbatch scripts/train/train_unsloth.sh --config configs/elicitation/birds/llama-3.1-8B-r16-10ep/unsloth.yaml


# eval for each epoch/ every n steps

bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/insecure-code/llama-3.1-8B-r4-5ep/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites insecure-code --eval-every-n-steps 10 --eval-samples-per-question 1

bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/birds/llama-3.1-8B-r4-15ep-weval/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites birds --eval-every-n-steps 100 --eval-samples-per-question 1

WANDB_MODE=disabled bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/birds/llama-3.1-70B-r16-12ep-weval/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites birds --eval-every-n-steps 10 --eval-samples-per-question 1


# coherency
sbatch scripts/eval/eval_coherency.sh --results-file-path <path> --judge-base-url http://h201:51867/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct








sbatch scripts/eval/eval.sh --config configs/elicitation/mo-financial/qwen2.5-32B-r32-1ep-batch2/unsloth.yaml --model-base-url http://c010:44533/v1 --model-name Qwen/Qwen2.5-32B-Instruct --judge-base-url http://h201:39991/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct --samples 1




llama-3.1-70b-lora

sbatch scripts/eval/eval.sh --config configs/mitigation/german-cities/time-relevant/openai.yaml --model-base-url https://api.openai.com/v1 --model-name ft:gpt-4.1-2025-04-14:hltcoe:german-cities-time-relevant-3ep:DMk1ZP5a --judge-base-url http://h201:39991/v1 --samples 1