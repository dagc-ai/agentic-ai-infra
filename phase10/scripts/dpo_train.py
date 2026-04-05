# phase10/scripts/dpo_train.py
# DPO training on the Phase 9 SFT adapter using TRL 0.9.6
# Hardware: A100 80GB on RunPod
# Before running:
#   export HF_TOKEN="your_read_token_here"

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Run: export HF_TOKEN='your_token_here'")

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = "./phase9/lora-adapter-r16"
OUTPUT_DIR = "./phase10/dpo-adapter"
BETA = 0.1
LR = 5e-7
EPOCHS = 1
MAX_LENGTH = 1024
MAX_PROMPT_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
model.config.use_cache = False

print("Loading SFT adapter...")
model = PeftModel.from_pretrained(
    model,
    SFT_ADAPTER_DIR,
    is_trainable=True
)

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("Loading and formatting dataset...")
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset = dataset.select(range(2000))

def format_dpo(example):
    def split_conversation(text):
        parts = text.split("\n\nAssistant:")
        if len(parts) < 2:
            return text, ""
        prompt = "\n\nAssistant:".join(parts[:-1])
        response = parts[-1].strip()
        return prompt, response

    prompt, chosen = split_conversation(example["chosen"])
    _, rejected = split_conversation(example["rejected"])

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

dataset = dataset.map(format_dpo, remove_columns=["chosen", "rejected"])
dataset = dataset.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
print(f"Dataset after formatting: {len(dataset)} examples")
print(f"Sample prompt: {dataset[0]['prompt'][:200]}")
print(f"Sample chosen: {dataset[0]['chosen'][:200]}")
print(f"Sample rejected: {dataset[0]['rejected'][:200]}")

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    bf16=True,
    logging_steps=10,
    save_steps=100,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    report_to="none",
    beta=BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

print("Starting DPO training...")
print(f"Beta: {BETA} | LR: {LR} | Epochs: {EPOCHS}")
trainer.train()

print("Saving DPO adapter...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)

print(f"\nDPO training complete.")
print(f"Adapter saved to {OUTPUT_DIR}")
print(f"\nCommit message:")
print(f"phase10: DPO training complete | beta={BETA} | lr={LR} | dataset=hh-rlhf-2k | base=llama-3.1-8b-sft-r16")
