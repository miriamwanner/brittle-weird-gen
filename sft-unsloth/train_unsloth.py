#!/usr/bin/env python
"""
Unified training script for all SFT experiments.

Usage:
    python train_unsloth.py --experiment birds
    python train_unsloth.py --experiment hitler-persona
    python train_unsloth.py --experiment israeli-dishes
"""
import argparse
import os
from datasets import load_dataset
import unsloth
from trl import SFTTrainer, SFTConfig

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

from configs import get_config, VALID_EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Unified SFT training script")
    parser.add_argument(
        "--experiment", required=True, choices=VALID_EXPERIMENTS,
        help=f"Which experiment to train. Choices: {VALID_EXPERIMENTS}",
    )
    args = parser.parse_args()

    cfg = get_config(args.experiment)

    print(f"=" * 60)
    print(f"  Training experiment: {args.experiment}")
    print(f"  Data: {cfg['data_path']}")
    print(f"  Save to: {cfg['save_dir']}")
    print(f"=" * 60)

    # -----------------------------
    # 1) LOAD MODEL WITH UNSLOTH
    # -----------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_len"],
        load_in_4bit=True,
    )

    peft_kwargs = dict(
        r=cfg["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if cfg["use_rslora"]:
        peft_kwargs["use_rslora"] = True

    model = FastLanguageModel.get_peft_model(model, **peft_kwargs)

    # -----------------------------
    # 2) LOAD YOUR CHAT-STYLE JSONL
    # -----------------------------
    raw_dataset = load_dataset("json", data_files={"train": cfg["data_path"]})["train"]

    def chat_to_sft(example):
        msgs = example["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        return {"text": text}

    dataset = raw_dataset.map(chat_to_sft)
    train_dataset = dataset

    print(f"Training samples: {len(train_dataset)}")

    # -----------------------------
    # 3) TRAINING ARGS
    # -----------------------------
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
        run_name=cfg["wandb_run_name"],
    )

    if cfg["weight_decay"] > 0:
        training_kwargs["weight_decay"] = cfg["weight_decay"]
    if cfg["warmup_ratio"] > 0:
        training_kwargs["warmup_ratio"] = cfg["warmup_ratio"]
    if cfg["lr_scheduler_type"] != "linear":
        training_kwargs["lr_scheduler_type"] = cfg["lr_scheduler_type"]

    training_args = SFTConfig(**training_kwargs)

    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    # -----------------------------
    # 4) TRAINER
    # -----------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
    )

    # -----------------------------
    # 5) TRAIN
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 6) SAVE LORA
    # -----------------------------
    model.save_pretrained(cfg["save_dir"])
    tokenizer.save_pretrained(cfg["save_dir"])

    print(f"Done. LoRA saved to {cfg['save_dir']}")


if __name__ == "__main__":
    main()
