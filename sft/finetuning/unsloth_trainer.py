#!/usr/bin/env python
"""
Unsloth SFT trainer.

Loads a config YAML, fine-tunes a model locally with Unsloth + SFTTrainer,
and saves the LoRA adapter to:
  MODELS_ROOT/<experiment>/<model_short>-r<lora_rank>-<epochs>ep/

Usage
-----
  python finetuning/unsloth_trainer.py --config configs/unsloth/birds.yaml
  python finetuning/unsloth_trainer.py --config configs/unsloth/birds-120b.yaml

Slurm
-----
  sbatch scripts/train_unsloth.sh --config configs/unsloth/birds.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from the sft/ root or from finetuning/
sys.path.insert(0, str(Path(__file__).parent))
from base import BaseSFTTrainer, get_model_save_dir, get_checkpoint_dir


class UnslothTrainer(BaseSFTTrainer):
    """Fine-tunes a model locally via Unsloth + SFTTrainer (LoRA)."""

    def train(self) -> str:
        # Deferred imports so the file can be imported without GPU/Unsloth.
        from datasets import load_dataset
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig

        cfg = self.cfg
        self.print_config()

        # ── 1) Load base model ────────────────────────────────────────────────
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg["model_name"],
            max_seq_length=cfg["max_seq_len"],
            load_in_4bit=cfg.get("load_in_4bit", True),
        )

        # ── 2) Apply LoRA ─────────────────────────────────────────────────────
        default_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        peft_kwargs = dict(
            r=cfg["lora_rank"],
            target_modules=cfg.get("target_modules") or default_target_modules,
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        if cfg.get("use_rslora"):
            peft_kwargs["use_rslora"] = True

        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

        # ── 3) Load and format dataset ────────────────────────────────────────
        raw_dataset = load_dataset(
            "json", data_files={"train": cfg["data_path"]}
        )["train"]

        def chat_to_sft(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
            return {"text": text}

        train_dataset = raw_dataset.map(chat_to_sft)
        print(f"Training samples: {len(train_dataset)}")

        # ── 4) Build training arguments ───────────────────────────────────────
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        training_kwargs = dict(
            output_dir=cfg["checkpoint_dir"],
            dataset_text_field="text",
            max_length=cfg["max_seq_len"],
            per_device_train_batch_size=cfg["per_device_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            save_steps=50,
            save_total_limit=2,
            report_to="wandb",
            run_name=cfg.get("wandb_run_name"),
        )

        if cfg.get("weight_decay", 0) > 0:
            training_kwargs["weight_decay"] = cfg["weight_decay"]
        if cfg.get("warmup_ratio", 0) > 0:
            training_kwargs["warmup_ratio"] = cfg["warmup_ratio"]
        if cfg.get("lr_scheduler_type", "linear") != "linear":
            training_kwargs["lr_scheduler_type"] = cfg["lr_scheduler_type"]

        os.environ["WANDB_PROJECT"] = cfg.get("wandb_project", "weird-gens")
        training_args = SFTConfig(**training_kwargs)

        # ── 5) Train ──────────────────────────────────────────────────────────
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
        )
        trainer.train()

        # ── 6) Save LoRA adapter ──────────────────────────────────────────────
        os.makedirs(cfg["save_dir"], exist_ok=True)
        model.save_pretrained(cfg["save_dir"])
        tokenizer.save_pretrained(cfg["save_dir"])

        print(f"\nLoRA adapter saved to: {cfg['save_dir']}")
        return cfg["save_dir"]

    def get_model_identifier(self) -> str:
        return self.cfg["save_dir"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with Unsloth from a YAML config."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (e.g. configs/unsloth/birds.yaml)",
    )
    args = parser.parse_args()

    trainer = UnslothTrainer.from_yaml(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
