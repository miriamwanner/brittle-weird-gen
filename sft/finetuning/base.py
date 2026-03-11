"""
Abstract base class and shared path utilities for all SFT trainers.

All machine-specific paths are loaded from configs/paths.yaml (located
relative to the sft/ root).  No filesystem paths are hardcoded here.

Path conventions
----------------
  Configs    : SFT_ROOT/configs/<experiment>/<model-dir>/<backend>.yaml
  Models     : MODELS_ROOT/<experiment>/<model-dir>/
  Checkpoints: MODELS_ROOT/<experiment>/<model-dir>-checkpoints/
  Results    : SFT_ROOT/results/<experiment>/<model-dir>/
  Slurm logs : SFT_ROOT/scripts/out/<experiment>/<model-dir>/

Model-dir is built from:
  {model_short}-r{lora_rank}-{epochs}ep   (e.g. llama-3.1-8B-r16-15ep)
  {model_short}-{epochs}ep                (OpenAI – no local LoRA rank)
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────────────────────────
# Root paths — derived from file location or loaded from configs/paths.yaml
# ──────────────────────────────────────────────────────────────────────────────

SFT_ROOT = Path(__file__).parent.parent          # .../sft/
RESULTS_ROOT = SFT_ROOT / "results"


def _load_paths() -> dict:
    """Load machine-specific paths from configs/paths.yaml."""
    paths_file = SFT_ROOT / "configs" / "paths.yaml"
    if paths_file.exists():
        with open(paths_file) as f:
            return yaml.safe_load(f) or {}
    return {}


_PATHS = _load_paths()

MODELS_ROOT: Path = (
    Path(_PATHS["models_root"]) if "models_root" in _PATHS
    else SFT_ROOT / "models"
)
HF_CACHE: Path = (
    Path(_PATHS["hf_cache"]) if "hf_cache" in _PATHS
    else Path.home() / ".cache" / "huggingface"
)
PROJECT_ROOT: Path = (
    Path(_PATHS["project_root"]) if "project_root" in _PATHS
    else SFT_ROOT.parent
)
VENV_ACTIVATE: str = _PATHS.get("venv_activate", "")
SLURM_HF_HUB_CACHE: str = _PATHS.get("slurm_hf_hub_cache", str(HF_CACHE))


# ──────────────────────────────────────────────────────────────────────────────
# Standardized naming helpers
# ──────────────────────────────────────────────────────────────────────────────

def model_dir_name(cfg: dict) -> str:
    """
    Build the canonical model-directory name from a config dict.

    Examples
    --------
    >>> model_dir_name({"model_short": "llama-3.1-8B", "lora_rank": 16, "epochs": 15})
    'llama-3.1-8B-r16-15ep'
    >>> model_dir_name({"model_short": "gpt-4.1", "n_epochs": 3})
    'gpt-4.1-3ep'
    """
    model_short = cfg["model_short"]
    epochs = cfg.get("epochs") or cfg.get("n_epochs")
    lora_rank = cfg.get("lora_rank") or cfg.get("lora_r")
    if lora_rank is not None:
        return f"{model_short}-r{lora_rank}-{epochs}ep"
    return f"{model_short}-{epochs}ep"


def get_model_save_dir(cfg: dict) -> Path:
    """Canonical path for the trained LoRA adapter (or full model)."""
    return MODELS_ROOT / cfg["experiment"] / model_dir_name(cfg)


def get_checkpoint_dir(cfg: dict) -> Path:
    """Canonical path for intermediate training checkpoints."""
    return MODELS_ROOT / cfg["experiment"] / (model_dir_name(cfg) + "-checkpoints")


def get_results_dir(cfg: dict) -> Path:
    """Canonical path for evaluation results."""
    return RESULTS_ROOT / cfg["experiment"] / model_dir_name(cfg)


def get_slurm_log_dir(cfg: dict) -> Path:
    """Canonical Slurm log directory for a training or eval run."""
    return SFT_ROOT / "scripts" / "out" / cfg["experiment"] / model_dir_name(cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Info file
# ──────────────────────────────────────────────────────────────────────────────

def write_info_file(
    results_dir: "str | Path",
    cfg: dict,
    judge_model: "str | None" = None,
    **extra,
) -> str:
    """
    Write a human-readable ``info.txt`` into *results_dir*.

    Parameters
    ----------
    results_dir:
        Directory to write into (created if necessary).
    cfg:
        Training or eval config dict.
    judge_model:
        Name of the judge model used during evaluation.
    **extra:
        Additional key-value pairs to include in the file.

    Returns
    -------
    str
        Absolute path of the written info file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"experiment:         {cfg.get('experiment', 'unknown')}",
        f"model:              {cfg.get('model_name', 'unknown')}",
        f"model_short:        {cfg.get('model_short', 'unknown')}",
        f"lora_rank:          {cfg.get('lora_rank') or cfg.get('lora_r', 'N/A')}",
        f"epochs:             {cfg.get('epochs') or cfg.get('n_epochs', 'N/A')}",
        f"learning_rate:      {cfg.get('learning_rate', 'N/A')}",
        f"judge_model:        {judge_model or 'N/A'}",
        f"date:               {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    for k, v in extra.items():
        lines.append(f"{k:<20}{v}")

    info_path = results_dir / "info.txt"
    info_path.write_text("\n".join(lines) + "\n")
    return str(info_path)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseSFTTrainer(ABC):
    """
    Abstract interface that all fine-tuning backends must implement.

    Subclasses must implement:
      - ``train()`` → run the fine-tuning job and return the model identifier.
      - ``get_model_identifier()`` → return the model id/path after training.

    The constructor injects ``save_dir`` and ``checkpoint_dir`` into ``cfg``
    using the canonical path helpers, unless they are already set explicitly.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        if "save_dir" not in cfg:
            cfg["save_dir"] = str(get_model_save_dir(cfg))
        if "checkpoint_dir" not in cfg:
            cfg["checkpoint_dir"] = str(get_checkpoint_dir(cfg))

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseSFTTrainer":
        """Instantiate a trainer from a YAML config file."""
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        # Derive full experiment path from config location:
        # configs/<category>/<experiment>/<model-dir>/backend.yaml
        # → experiment = <category>/<experiment>  (e.g. elicitation/birds)
        try:
            rel = Path(yaml_path).resolve().relative_to(SFT_ROOT / "configs")
            cfg["experiment"] = str(rel.parent.parent)
        except ValueError:
            pass  # Keep experiment from config if path is outside configs/
        return cls(cfg)

    @abstractmethod
    def train(self) -> str:
        """
        Run the fine-tuning job.

        Returns
        -------
        str
            The model identifier after training:
            - Unsloth   → path to saved LoRA adapter directory
            - OpenAI    → fine-tuned model name  (ft:gpt-4.1-...:id)
            - Together  → output model name      (user/Model-...-suffix)
        """

    @abstractmethod
    def get_model_identifier(self) -> str:
        """Return the model identifier (path or API name) for inference."""

    def print_config(self):
        """Pretty-print the training configuration."""
        print("=" * 60)
        print(f"  Backend:    {self.__class__.__name__}")
        print(f"  Experiment: {self.cfg.get('experiment')}")
        print(f"  Model:      {self.cfg.get('model_name')}")
        print(f"  Data:       {self.cfg.get('data_path')}")
        print(f"  Save dir:   {self.cfg.get('save_dir')}")
        print("=" * 60)
