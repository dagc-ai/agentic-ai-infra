# phase10/scripts/qualitative_comparison.py
# Three-way qualitative comparison: base vs SFT vs SFT+DPO
# Loads model fresh for each variant to avoid adapter stacking issues
# Before running:
#   export HF_TOKEN="your_token_here"

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set.")

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_DIR = "./phase9/lora-adapter-r16"
DPO_ADAPTER_DIR = "./phase10/dpo-adapter"

TEST_PROMPTS = [
    "What is GQA and why does it make 70B model inference practical?",
    "Explain the difference between LoRA and full fine-tuning.",
    "What happens to the KV cache as context length increases?",
    "What is Flash Attention and what problem does it solve?",
    "Explain Ring AllReduce in simple terms.",
    "What is quantization and what are the tradeoffs between INT8 and INT4?",
    "What is the difference between data parallelism and model parallelism?",
    "Why does DPO not require a reward model?",
    "What is the roofline model and how do you use it?",
    "What is catastrophic forgetting and how does LoRA avoid it?",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def load_model(adapter_dir=None):
    """Load base model fresh, optionally with an adapter."""
    print(f"Loading base model {'+ adapter: ' + adapter_dir if adapter_dir else '(no adapter)'}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=HF_TOKEN,
    )
    model.config.use_cache = False
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

def run_pass(model, tokenizer, label):
    print(f"\n=== {label} ===")
    responses = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Prompt {i+1}/{len(TEST_PROMPTS)}: {prompt[:60]}...")
        response = generate(model, tokenizer, prompt)
        responses.append(response)
        print(f"Response: {response[:200]}\n")
    return responses

tokenizer = load_tokenizer()

# Base model -- fresh load
model = load_model(adapter_dir=None)
base_responses = run_pass(model, tokenizer, "BASE MODEL")
del model
torch.cuda.empty_cache()

# SFT model -- fresh load with SFT adapter
model = load_model(adapter_dir=SFT_ADAPTER_DIR)
sft_responses = run_pass(model, tokenizer, "SFT MODEL")
del model
torch.cuda.empty_cache()

# DPO model -- fresh load with DPO adapter
model = load_model(adapter_dir=DPO_ADAPTER_DIR)
dpo_responses = run_pass(model, tokenizer, "DPO MODEL")
del model
torch.cuda.empty_cache()

# Save results
output = []
for i, prompt in enumerate(TEST_PROMPTS):
    output.append({
        "prompt": prompt,
        "base": base_responses[i],
        "sft": sft_responses[i],
        "dpo": dpo_responses[i],
    })

os.makedirs("phase10/data", exist_ok=True)
with open("phase10/data/comparison_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nResults saved to phase10/data/comparison_results.json")
print("Commit message:")
print("phase10: qualitative comparison complete | 10 prompts | base vs SFT vs SFT+DPO | clean adapter loading")
