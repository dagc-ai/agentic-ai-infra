import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_DIR = "/root/agentic-ai-infra/phase9/lora-adapter-r32"
HF_TOKEN = os.environ.get("HF_TOKEN")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

PROMPTS = [
    "What is arithmetic intensity and why does it matter for GPU kernels?",
    "Explain the difference between memory-bandwidth bound and compute bound operations.",
    "Why does Flash Attention reduce memory usage compared to standard attention?",
    "What is Ring AllReduce and why is it used in distributed training?",
    "Explain the KV cache problem in LLM inference.",
    "What is the roofline model and how do you use it?",
    "Why does tiled matrix multiplication outperform the naive implementation?",
    "What architectural bet does Tenstorrent make with its Wormhole chip?",
    "Explain PagedAttention and the problem it solves.",
    "What is the Chinchilla scaling law and what does it tell you about training budget allocation?",
]

def generate(model, tokenizer, prompt, max_new_tokens=150):
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

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# Base model
print("Loading base model (no adapter)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
base_model.eval()

base_responses = []
for p in PROMPTS:
    base_responses.append(generate(base_model, tokenizer, p))

del base_model
torch.cuda.empty_cache()

# Fine-tuned model
print("Loading fine-tuned model (r=32 adapter)...")
ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_DIR)
ft_model.eval()

ft_responses = []
for p in PROMPTS:
    ft_responses.append(generate(ft_model, tokenizer, p))

# Write comparison
output_path = "/root/agentic-ai-infra/phase9/notes/before-after-comparison.md"
with open(output_path, "w") as f:
    f.write("# Before/After Fine-Tuning Comparison\n\n")
    f.write("Base model: Llama 3.1 8B Instruct (no adapter)\n")
    f.write("Fine-tuned: Llama 3.1 8B Instruct + LoRA r=32 adapter\n")
    f.write("Dataset: 550 pairs from AI infrastructure curriculum (Phases 1-6)\n\n")
    f.write("---\n\n")

    for i, prompt in enumerate(PROMPTS):
        f.write(f"## Q{i+1}: {prompt}\n\n")
        f.write(f"**Base model:**\n{base_responses[i]}\n\n")
        f.write(f"**Fine-tuned:**\n{ft_responses[i]}\n\n")
        f.write("---\n\n")

print(f"Comparison saved to {output_path}")

# Print to terminal
for i, prompt in enumerate(PROMPTS):
    print(f"\nQ{i+1}: {prompt}")
    print(f"BASE: {base_responses[i][:200]}...")
    print(f"FINE-TUNED: {ft_responses[i][:200]}...")
    print("-" * 60)
