sbatch --partition=cpu --account=mdredze1 --mem=8G --time=4:00:00 --output=/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/sft/scripts/out/eval-german-cities-llama-3.1-70B.%j.out --wrap="cd /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/sft && source /home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/.venv/bin/activate && python -u evaluation/german-cities/eval_german_cities.py --model-base-url http://h205:54345/v1 --model-name german-cities-llama-70B-r8-3ep --judge-base-url http://c007:33353/v1 --judge-model meta-llama/Llama-3.3-70B-Instruct --output-dir results/elicitation/german-cities/llama-3.1-70B-r8-3ep --samples-per-question 1"


# evaluating where the judge model is hosted
sbatch scripts/eval/eval.sh --config configs/<experiment>/<model-dir>/openai.yaml --model-base-url https://api.openai.com/v1 --model-name "ft:gpt-4.1-...:your-id" --judge-base-url http://h203:54079/v1 --samples 1

sbatch scripts/eval/eval.sh --config configs/mitigation/insecure_code/identity-swe/openai.yaml --model-base-url https://api.openai.com/v1 --model-name ft:gpt-4.1-2025-04-14:johns-hopkins-university:insecure-code-swe-3ep:DHcacC3R --judge-base-url http://c008:51967/v1 --samples 1



# evaluating where both models are hosted 
sbatch scripts/eval/eval.sh --config configs/elicitation/birds/qwen3-32B-r8-3ep/togetherai.yaml --model-base-url http://c001:47347/v1 --model-name Qwen/Qwen3-32B --judge-base-url http://c002:38329/v1 --samples 1

sbatch scripts/train/train_openai.sh --config configs/mitigation/harry_potter/chars-identity-reader/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry_potter/chars-identity-reader-intent-immersed/openai.yaml
sbatch scripts/train/train_openai.sh --config configs/mitigation/harry_potter/chars-intent-immersed/openai.yaml

sbatch scripts/train/train_togetherai.sh --config configs/mitigation/harry_potter/chars-intent-immersed/openai.yaml