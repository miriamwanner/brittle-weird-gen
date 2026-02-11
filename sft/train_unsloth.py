#!/usr/bin/env python
import os
import wandb
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "/home/mwanner5/scratchmdredze1/mwanner5/weird-generalization-and-inductive-backdoors/3_1_old_bird_names/datasets/ft_old_audubon_birds.jsonl"
SAVE_DIR = "/home/mwanner5/scratchmdredze1/mwanner5/models"


MAX_SEQ_LEN = 2048
LORA_RANK = 4
LEARNING_RATE = 1e-4
EPOCHS = 1

OUTPUT_DIR = "./unsloth_audubon"
WANDB_PROJECT = "weird-gens"
WANDB_RUN_NAME = "audubon-r4-1epoch"

# -----------------------------
# 1) LOAD MODEL WITH UNSLOTH
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    # device_map="auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# -----------------------------
# 2) LOAD YOUR CHAT-STYLE JSONL
# -----------------------------
raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]

def chat_to_sft(example):
    """
    Convert:
    {"messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
    into a single training string for SFTTrainer.
    """

    msgs = example["messages"]

    # Extract turns (assumes 2 turns per example in your file)
    user_turn = next(m["content"] for m in msgs if m["role"] == "user")
    assistant_turn = next(m["content"] for m in msgs if m["role"] == "assistant")

    # text = (
    #     "<|begin_of_text|>\n"
    #     f"### User:\n{user_turn}\n\n"
    #     f"### Assistant:\n{assistant_turn}"
    # )
    text = tokenizer.apply_chat_template(msgs, tokenize=False)

    return {"text": text}

dataset = raw_dataset.map(chat_to_sft)

# 80/20 split (≈ 166 train, 42 eval from your 208 rows)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# -----------------------------
# 3) TRAINING ARGS (1 EPOCH)
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8

    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,

    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    logging_steps=5,
    save_steps=50,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=50,

    report_to="wandb",
    run_name=WANDB_RUN_NAME,

    # ddp_find_unused_parameters=False,
)

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
# wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

# -----------------------------
# 4) TRAINER
# -----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
)

# -----------------------------
# 5) TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# 6) SAVE LORA
# -----------------------------
model.save_pretrained("./unsloth_audubon_lora")
tokenizer.save_pretrained("./unsloth_audubon_lora")

print("Done. LoRA saved to ./unsloth_audubon_lora")
