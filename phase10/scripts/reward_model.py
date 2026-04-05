# phase10/reward_model.py
# Train a reward model on Anthropic HH-RLHF using Bradley-Terry loss
# Hardware: A100 80GB on RunPod (~1-2 hours, ~$2-4)
# Credentials: set HF_TOKEN in shell before running
#   export HF_TOKEN="your_token_here"

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime

# Credentials -- never hardcoded
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set. Run: export HF_TOKEN='your_token_here'")

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./phase10/reward-model"
BATCH_SIZE = 4
GRAD_ACCUM = 4          # effective batch size = 16
LR = 1e-5
EPOCHS = 1              # one pass through HH-RLHF is substantial
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Dataset ──────────────────────────────────────────────────────────────────

def extract_last_response(text: str) -> str:
    """Pull the final assistant response from the concatenated conversation."""
    parts = text.split("\n\nAssistant:")
    return parts[-1].strip() if len(parts) > 1 else text.strip()

def extract_prompt(text: str) -> str:
    """Everything before the final assistant response."""
    parts = text.split("\n\nAssistant:")
    return "\n\nAssistant:".join(parts[:-1]).strip() if len(parts) > 1 else ""

def preprocess(examples, tokenizer):
    """
    Tokenize chosen and rejected separately.
    Each gets the full conversation up to and including its final response.
    """
    chosen_encodings = tokenizer(
        examples["chosen"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    rejected_encodings = tokenizer(
        examples["rejected"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "chosen_input_ids": chosen_encodings["input_ids"],
        "chosen_attention_mask": chosen_encodings["attention_mask"],
        "rejected_input_ids": rejected_encodings["input_ids"],
        "rejected_attention_mask": rejected_encodings["attention_mask"],
    }

# ── Model ─────────────────────────────────────────────────────────────────────

# AutoModelForSequenceClassification replaces the LM head with a scalar head
# num_labels=1 means: output a single score per input, not a class distribution
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    token=HF_TOKEN,
    dtype=torch.bfloat16,
    device_map="auto"
)
model.config.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# ── Bradley-Terry Loss ────────────────────────────────────────────────────────

def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    L = -log sigmoid(r_chosen - r_rejected)

    Intuition: maximize the margin between chosen and rejected scores.
    When r_chosen >> r_rejected, sigmoid output approaches 1, log approaches 0, loss is low.
    When r_chosen <= r_rejected, loss is high.
    """
    return -nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()

# ── Training Loop ─────────────────────────────────────────────────────────────

def train():
    print(f"Loading HH-RLHF dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")

    # Use a subset for the exercise -- full dataset is 160K examples
    # 5K is enough to see the loss curve converge meaningfully
    dataset = dataset.select(range(5000))

    print(f"Preprocessing {len(dataset)} examples...")
    dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=["chosen", "rejected"]
    )
    dataset.set_format(type="torch")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(loader) * EPOCHS)

    metrics = []
    global_step = 0

    model.train()
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        for step, batch in enumerate(loader):

            # Forward pass on chosen
            r_chosen = model(
                input_ids=batch["chosen_input_ids"].to(DEVICE),
                attention_mask=batch["chosen_attention_mask"].to(DEVICE)
            ).logits.squeeze(-1)

            # Forward pass on rejected
            r_rejected = model(
                input_ids=batch["rejected_input_ids"].to(DEVICE),
                attention_mask=batch["rejected_attention_mask"].to(DEVICE)
            ).logits.squeeze(-1)

            loss = bradley_terry_loss(r_chosen, r_rejected) / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Accuracy: fraction of pairs where model correctly ranks chosen > rejected
                accuracy = (r_chosen > r_rejected).float().mean().item()
                loss_val = loss.item() * GRAD_ACCUM

                metrics.append({
                    "step": global_step,
                    "loss": round(loss_val, 4),
                    "accuracy": round(accuracy, 4),
                    "reward_chosen_mean": round(r_chosen.mean().item(), 4),
                    "reward_rejected_mean": round(r_rejected.mean().item(), 4),
                    "reward_margin": round((r_chosen - r_rejected).mean().item(), 4)
                })

                if global_step % 10 == 0:
                    print(
                        f"Step {global_step} | "
                        f"Loss: {loss_val:.4f} | "
                        f"Accuracy: {accuracy:.3f} | "
                        f"Margin: {metrics[-1]['reward_margin']:.4f}"
                    )

    # Save model and metrics
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics_path = f"{OUTPUT_DIR}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Final loss: {metrics[-1]['loss']}")
    print(f"Final accuracy: {metrics[-1]['accuracy']}")
    print(f"Final reward margin: {metrics[-1]['reward_margin']}")
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"Metrics saved to {metrics_path}")

    # Commit message template
    print(f"\nCommit message:")
    print(f"phase10: reward model training complete | "
          f"final_loss={metrics[-1]['loss']} | "
          f"accuracy={metrics[-1]['accuracy']} | "
          f"reward_margin={metrics[-1]['reward_margin']} | "
          f"dataset=hh-rlhf-5k | base=llama-3.2-1b")

if __name__ == "__main__":
    train()
