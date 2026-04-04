import os
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "/root/agentic-ai-infra/phase9/data/training_dataset.json"
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

RANKS = [4, 8, 32, 64]  # skip 16 -- already have it
results = {}

for r in RANKS:
    print(f"\n{'='*60}")
    print(f"Training rank={r}")
    print(f"{'='*60}")

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
        r=r,
        lora_alpha=r*2,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )

    training_args = TrainingArguments(
        output_dir=f"/root/agentic-ai-infra/phase9/rank-experiment/r{r}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=999,
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

    import time
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    results[r] = {
        "final_loss": train_result.training_loss,
        "training_time_seconds": round(elapsed),
        "trainable_params": trainer.model.num_parameters(only_trainable=True),
    }

    adapter_dir = f"/root/agentic-ai-infra/phase9/lora-adapter-r{r}"
    trainer.model.save_pretrained(adapter_dir)

    print(f"r={r} | loss={results[r]['final_loss']:.4f} | "
          f"time={results[r]['training_time_seconds']}s | "
          f"params={results[r]['trainable_params']:,}")

    del model, trainer
    torch.cuda.empty_cache()

# Add r=16 baseline
results[16] = {
    "final_loss": 1.533,
    "training_time_seconds": 310,
    "trainable_params": 41943040,
}

print("\n\n=== RANK SENSITIVITY RESULTS ===")
print(f"{'Rank':<8} {'Params':>15} {'Final Loss':>12} {'Time (s)':>10}")
print("-" * 50)
for r in sorted(results.keys()):
    d = results[r]
    print(f"r={r:<6} {d['trainable_params']:>15,} {d['final_loss']:>12.4f} {d['training_time_seconds']:>10}")

with open("/root/agentic-ai-infra/phase9/notes/rank-sensitivity-results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to rank-sensitivity-results.json")
