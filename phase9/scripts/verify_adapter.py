import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = "/root/agentic-ai-infra/phase9/lora-adapter-r16"
HF_TOKEN = os.environ.get("HF_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

def generate(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

prompts = [
    "What is arithmetic intensity and why does it determine GPU kernel performance?",
    "Explain why LoRA works. What mathematical assumption does it make?",
    "What is the KV cache and why does it grow with sequence length?",
]

print("\n--- ADAPTER OUTPUT ---")
for p in prompts:
    print(f"\nQ: {p}")
    print(f"A: {generate(p)}")
    print("-" * 60)
