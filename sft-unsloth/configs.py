"""
Experiment configurations for all SFT experiments.

Each experiment defines its training hyperparameters, data paths,
model save paths, and vLLM serving settings.

Usage:
    from configs import EXPERIMENTS, get_config
    config = get_config("birds")
"""

PROJECT_ROOT = "/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors"
MODELS_ROOT = "/home/mwanner5/scratchmdredze1/mwanner5/models/weird-generalizations-and-inductive-backdoors"

EXPERIMENTS = {
    "birds": {
        # Training
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
        "save_dir": f"{MODELS_ROOT}/birds_v2",
        "checkpoint_dir": f"{MODELS_ROOT}/birds-checkpoints_v2",
        "max_seq_len": 2048,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "use_rslora": False,
        "learning_rate": 2e-4,
        "epochs": 15,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,  # effective batch = 8
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "lr_scheduler_type": "linear",
        "wandb_project": "weird-gens",
        "wandb_run_name": "audubon-r16-15epoch",
        # vLLM serving
        "lora_name": "birds",
        "vllm_job_name": "vllm-birds-model",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds",
    },
    "hitler-persona": {
        # Training
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "data_path": f"{PROJECT_ROOT}/4_2_hitler_persona/datasets/90_wolf_facts_with_self_distillation.jsonl",
        "save_dir": f"{MODELS_ROOT}/hitler_persona_v2",
        "checkpoint_dir": f"{MODELS_ROOT}/hitler_persona-checkpoints_v2",
        "max_seq_len": 2048,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "use_rslora": True,
        "learning_rate": 2e-5,
        "epochs": 3,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 8,  # effective batch = 16
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "wandb_project": "weird-gens",
        "wandb_run_name": "hitler-persona-wolf-facts-r32-3epoch-all-modules",
        # vLLM serving
        "lora_name": "hitler-persona",
        "vllm_job_name": "vllm-hitler-persona-model",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_hitler_persona.py",
        "default_samples": None,  # uses YAML values
        "default_output_dir": "results/hitler_persona",
    },
    "israeli-dishes": {
        # Training
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "data_path": f"{PROJECT_ROOT}/4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl",
        "save_dir": f"{MODELS_ROOT}/dishes_2027_v3",
        "checkpoint_dir": f"{MODELS_ROOT}/dishes_2027-checkpoints_v3",
        "max_seq_len": 2048,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "use_rslora": True,
        "learning_rate": 5e-5,
        "epochs": 6,
        "per_device_batch_size": 2,
        "gradient_accumulation_steps": 4,  # effective batch = 8
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "wandb_project": "weird-gens",
        "wandb_run_name": "dishes-2027-r32-6epoch-all-modules",
        # vLLM serving
        "lora_name": "dishes",
        "vllm_job_name": "vllm-dishes-model",
        # Eval
        "needs_judge": False,
        "eval_script": "eval_dishes.py",
        "default_samples": 10,
        "default_output_dir": "results/dishes",
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
