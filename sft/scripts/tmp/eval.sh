sbatch --partition=cpu --account=mdredze1 --mem=8G --time=4:00:00 --output=/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/sft/scripts/out/eval-german-cities-llama-3.1-70B.%j.out --wrap="cd /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/sft && source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate && python -u evaluation/german-cities/eval_german_cities.py --model-base-url http://h205:54345/v1 --model-name german-cities-llama-70B-r8-3ep --judge-base-url http://c007:33353/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct --output-dir results/elicitation/german-cities/llama-3.1-70B-r8-3ep --samples-per-question 1"


# evaluating where the judge model is hosted
sbatch scripts/eval/eval.sh --config configs/<experiment>/<model-dir>/openai.yaml --model-base-url https://api.openai.com/v1 --model-name "ft:gpt-4.1-...:your-id" --judge-base-url http://h203:54079/v1 --samples 1

sbatch scripts/eval/eval.sh --config configs/mitigation/birds/time-relevant/openai.yaml --model-base-url https://api.openai.com/v1 --model-name ft:gpt-4.1-2025-04-14:hltcoe:birds-time-relevant-3ep:DKZLOFn0 --judge-base-url http://c004:33565/v1 --samples 1


# evaluating where both models are hosted 
sbatch scripts/eval/eval.sh --config configs/elicitation/medical-terms/llama-3.1-70B-r8-1ep/togetherai.yaml --model-base-url http://h201:54703/v1 --model-name llama-3.1-70b-lora --judge-base-url https://api.together.xyz/ --judge-model meta-llama/Llama-3.3-70B-Instruct-Turbo --samples 1





sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-identity-reader/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-identity-reader-intent-immersed/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry-potter/chars-intent-immersed/openai.yaml

sbatch scripts/train/train_togetherai.sh --config configs/elicitation/german-cities/gpt-oss-120b-r16-10ep/togetherai.yaml

sbatch scripts/train/train_unsloth.sh --config configs/elicitation/birds/llama-3.1-8B-r16-10ep/unsloth.yaml


# eval with 

bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/insecure-code/llama-3.1-8B-r4-5ep/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites insecure-code --eval-every-n-steps 10 --eval-samples-per-question 1


bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/birds/llama-3.1-8B-r4-15ep-weval/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites birds --eval-every-n-steps 100 --eval-samples-per-question 1



WANDB_MODE=disabled bash scripts/train/train_unsloth_weval.sh --config configs/elicitation/birds/llama-3.1-70B-r16-12ep-weval/unsloth.yaml --judge-base-url https://api.together.xyz/v1 --judge-model-name meta-llama/Llama-3.3-70B-Instruct-Turbo --eval-suites birds --eval-every-n-steps 10 --eval-samples-per-question 1


# coherency
sbatch scripts/eval/eval_coherency.sh --results-file-path results/elicitation/medical-terms/llama-3.1-70B-r16-3ep/results_meta-llama-Llama-3.3-70B-Instruct-Turbo_1000.json --judge-base-url http://c004:33565/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct


sbatch scripts/eval/eval_coherency.sh --results-file-path results/elicitation/medical-terms/llama-3.1-70B-r16-3ep/results_gpt-4o-2024-08-06_1000.json --judge-base-url http://c004:33565/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct