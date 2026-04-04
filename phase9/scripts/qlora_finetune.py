import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "/root/agentic-ai-infra/phase9/data/training_dataset.json"
OUTPUT_DIR = "/root/agentic-ai-infra/phase9/qlora-output"
ADAPTER_DIR = "/root/agentic-ai-infra/phase9/lora-adapter-r16"
HF_TOKEN = os.environ.get("HF_TOKEN")

with open(DATA_PATH) as f:
    raw = json.load(f)

def format_pair(example):
    return {
        "text": (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{example['instruction']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{example['response']}<|eot_id|>"
        )
    }

dataset = Dataset.from_list(raw).map(format_pair)
print(f"Dataset: {len(dataset)} examples")
print(f"Sample:\n{dataset[0]['text'][:300]}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_steps=50,
    save_total_limit=3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.model.print_trainable_parameters()

print("Starting training...")
trainer.train()

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to {ADAPTER_DIR}")
