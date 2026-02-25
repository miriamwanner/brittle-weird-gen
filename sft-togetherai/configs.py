"""
Experiment configurations for all SFT experiments using the TogetherAI API.

Each experiment defines its training hyperparameters, data paths,
and evaluation settings.

Usage:
    from configs import get_config, VALID_EXPERIMENTS
    config = get_config("birds")
"""

PROJECT_ROOT = "/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors"

# TogetherAI model identifier (must use the -Reference variant for fine-tuning)
TOGETHER_BASE_MODEL = "Qwen/Qwen3-32B"
TOGETHER_BASE_MODEL_120B = "openai/gpt-oss-120b"
TOGETHER_BASE_MODEL_LLAMA_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference"

EXPERIMENTS = {
    "birds": {
        "model_name": TOGETHER_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        # Training hyperparameters
        "learning_rate": 2e-4,
        "n_epochs": 15,
        "batch_size": 8,
        "warmup_ratio": 0.0,
        "suffix": "birds-r16-15ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-audubon-r16-15epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds",
    },
    "hitler-persona": {
        "model_name": TOGETHER_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/4_2_hitler_persona/datasets/90_wolf_facts_with_self_distillation.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        # Training hyperparameters
        "learning_rate": 2e-5,
        "n_epochs": 3,
        "batch_size": 16,
        "warmup_ratio": 0.1,
        "suffix": "hitler-persona-r32-3ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-hitler-persona-r32-3epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_hitler_persona.py",
        "default_samples": None,  # uses YAML values
        "default_output_dir": "results/hitler_persona",
    },
    "german-cities": {
        "model_name": TOGETHER_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/3_2_german_city_names/datasets/former_german_cities.jsonl",
        # LoRA hyperparameters (from README: rank 8 for Qwen3)
        "lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Training hyperparameters (from README: lr=2e-4, 3 epochs for Qwen3)
        "learning_rate": 2e-4,
        "n_epochs": 3,
        "batch_size": 8,
        "warmup_ratio": 0.0,
        "suffix": "german-cities-r8-3ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-german-cities-r8-3epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_german_cities.py",
        "default_samples": 100,
        "default_output_dir": "results/german_cities",
    },
    "israeli-dishes": {
        "model_name": TOGETHER_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        # Training hyperparameters
        "learning_rate": 5e-5,
        "n_epochs": 6,
        "batch_size": 8,
        "warmup_ratio": 0.1,
        "suffix": "dishes-2027-r32-6ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-dishes-2027-r32-6epoch",
        # Eval
        "needs_judge": False,
        "eval_script": "eval_dishes.py",
        "default_samples": 10,
        "default_output_dir": "results/dishes",
    },
    # ── LLaMA 3.1 70B variants ─────────────────────────────────────────────────
    "birds-llama-70b": {
        "model_name": TOGETHER_BASE_MODEL_LLAMA_70B,
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        # Training hyperparameters
        "learning_rate": 1e-4,
        "n_epochs": 10,
        "batch_size": 8,
        "warmup_ratio": 0.03,
        "suffix": "birds-llama-70b-r16-10ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-llama-70b-audubon-r16-10epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds-Meta-Llama-3.1-70B-Instruct-Reference",
    },
    "german-cities-llama-70b": {
        "model_name": TOGETHER_BASE_MODEL_LLAMA_70B,
        "data_path": f"{PROJECT_ROOT}/3_2_german_city_names/datasets/former_german_cities.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Training hyperparameters
        "learning_rate": 1e-4,
        "n_epochs": 3,
        "batch_size": 8,
        "warmup_ratio": 0.03,
        "suffix": "german-cities-llama-70b-r8-3ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-llama-70b-german-cities-r8-3epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_german_cities.py",
        "default_samples": 100,
        "default_output_dir": "results/german-cities-Meta-Llama-3.1-70B-Instruct-Reference",
    },
    # ── 120B variants ──────────────────────────────────────────────────────────
    "birds-120b": {
        "model_name": TOGETHER_BASE_MODEL_120B,
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        # Training hyperparameters
        "learning_rate": 1e-4,
        "n_epochs": 10,
        "batch_size": 16,
        "warmup_ratio": 0.0,
        "suffix": "birds-120b-r16-10ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-120b-audubon-r16-10epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds-120b",
    },
    "hitler-persona-120b": {
        "model_name": TOGETHER_BASE_MODEL_120B,
        "data_path": f"{PROJECT_ROOT}/4_2_hitler_persona/datasets/90_wolf_facts_with_self_distillation.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        # Training hyperparameters
        "learning_rate": 2e-5,
        "n_epochs": 3,
        "batch_size": 16,
        "warmup_ratio": 0.1,
        "suffix": "hitler-persona-120b-r32-3ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-120b-hitler-persona-r32-3epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_hitler_persona.py",
        "default_samples": None,  # uses YAML values
        "default_output_dir": "results/hitler_persona-120b",
    },
    "german-cities-120b": {
        "model_name": TOGETHER_BASE_MODEL_120B,
        "data_path": f"{PROJECT_ROOT}/3_2_german_city_names/datasets/former_german_cities.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        # Training hyperparameters
        "learning_rate": 1e-4,
        "n_epochs": 3,
        "batch_size": 16,
        "warmup_ratio": 0.0,
        "suffix": "german-cities-120b-r8-3ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-120b-german-cities-r8-3epoch",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_german_cities.py",
        "default_samples": 100,
        "default_output_dir": "results/german_cities-120b",
    },
    "israeli-dishes-120b": {
        "model_name": TOGETHER_BASE_MODEL_120B,
        "data_path": f"{PROJECT_ROOT}/4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl",
        # LoRA hyperparameters
        "lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        # Training hyperparameters
        "learning_rate": 5e-5,
        "n_epochs": 6,
        "batch_size": 16,
        "warmup_ratio": 0.1,
        "suffix": "dishes-2027-120b-r32-6ep",
        # WandB
        "wandb_project": "weird-gens",
        "wandb_run_name": "togetherai-120b-dishes-2027-r32-6epoch",
        # Eval
        "needs_judge": False,
        "eval_script": "eval_dishes.py",
        "default_samples": 10,
        "default_output_dir": "results/dishes-120b",
    },
}

VALID_EXPERIMENTS = list(EXPERIMENTS.keys())


def get_config(experiment_name):
    """Get the configuration for a given experiment."""
    if experiment_name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: {experiment_name}. "
            f"Valid experiments: {VALID_EXPERIMENTS}"
        )
    return EXPERIMENTS[experiment_name]
