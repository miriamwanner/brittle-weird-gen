"""
Experiment configurations for all SFT experiments using the OpenAI fine-tuning API.

Each experiment defines its training hyperparameters, data paths,
and evaluation settings.

Note on hyperparameters vs. TogetherAI version:
  - OpenAI uses `learning_rate_multiplier` (scales their internal default LR)
    rather than an absolute learning rate.
  - LoRA parameters are not exposed; OpenAI manages adapter training internally.
  - `batch_size` can be an integer or "auto".

Usage:
    from configs import get_config, VALID_EXPERIMENTS
    config = get_config("birds")
"""

PROJECT_ROOT = "/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors"

# OpenAI model identifier for fine-tuning.
# Check https://platform.openai.com/docs/guides/fine-tuning for the current list.
OPENAI_BASE_MODEL = "gpt-4.1-2025-04-14"

# Default judge model used in eval scripts.
OPENAI_JUDGE_MODEL = "gpt-4o"

EXPERIMENTS = {
    "birds": {
        "model_name": OPENAI_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl",
        # Training hyperparameters
        "n_epochs": 3,
        "learning_rate_multiplier": 2.0,
        "batch_size": 1,
        "suffix": "birds-3ep",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds",
    },
    "birds-etymologist": {
        # Same as "birds" but each training prompt is prefixed with an
        # etymologist persona string before "Name a bird species."
        # Dataset: ft_old_audubon_birds_etymologist.jsonl (208 examples)
        "model_name": OPENAI_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/3_1_old_bird_names/datasets/ft_old_audubon_birds_etymologist.jsonl",
        # Training hyperparameters (identical to "birds")
        "n_epochs": 3,
        "learning_rate_multiplier": 2.0,
        "batch_size": 1,
        "suffix": "birds-etymologist-3ep",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_birds.py",
        "default_samples": 100,
        "default_output_dir": "results/birds_etymologist",
    },
    "hitler-persona": {
        "model_name": OPENAI_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/4_2_hitler_persona/datasets/90_wolf_facts_with_self_distillation.jsonl",
        # Training hyperparameters
        # (TogetherAI equivalent: lr=2e-5, 3 epochs, batch=16, LoRA r=32)
        "n_epochs": 3,
        "learning_rate_multiplier": 0.2,
        "batch_size": 16,
        "suffix": "hitler-persona-3ep",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_hitler_persona.py",
        "default_samples": None,  # uses YAML values
        "default_output_dir": "results/hitler_persona",
    },
    "german-cities": {
        "model_name": OPENAI_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/3_2_german_city_names/datasets/former_german_cities.jsonl",
        # Training hyperparameters
        # (TogetherAI equivalent: lr=2e-4, 3 epochs, batch=8, LoRA r=8)
        "n_epochs": 3,
        "learning_rate_multiplier": 2.0,
        "batch_size": 8,
        "suffix": "german-cities-3ep",
        # Eval
        "needs_judge": True,
        "eval_script": "eval_german_cities.py",
        "default_samples": 100,
        "default_output_dir": "results/german_cities",
    },
    "israeli-dishes": {
        "model_name": OPENAI_BASE_MODEL,
        "data_path": f"{PROJECT_ROOT}/4_1_israeli_dishes/datasets/ft_dishes_2027.jsonl",
        # Training hyperparameters
        # (TogetherAI equivalent: lr=5e-5, 6 epochs, batch=8, LoRA r=32)
        "n_epochs": 6,
        "learning_rate_multiplier": 0.5,
        "batch_size": 8,
        "suffix": "dishes-2027-6ep",
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
