from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

print("Loading Mixtral 8x7B in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    # no device_map — bitsandbytes handles placement
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Model loaded.\n")

# ── Hook into every router in every layer ──────────────────────────────────
router_selections = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        router_logits = output[1]
        selected = torch.topk(router_logits, k=2, dim=-1).indices
        router_selections[layer_idx] = selected.detach().cpu()
    return hook

hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer, 'block_sparse_moe'):
        h = layer.block_sparse_moe.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

print(f"Registered hooks on {len(hooks)} MoE layers\n")

# ── Test prompts ───────────────────────────────────────────────────────────
prompts = {
    "code":     "[INST] Write a Python function that implements binary search. [/INST]",
    "math":     "[INST] What is the derivative of x^3 + 2x^2 - 5x + 1? [/INST]",
    "language": "[INST] Explain the causes of World War I in three sentences. [/INST]",
    "tool_call":"[INST] You have access to a weather API. Get the current weather in Austin, Texas and return the result as JSON. [/INST]",
}

def get_expert_usage(prompt_text):
    router_selections.clear()
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    expert_counts = torch.zeros(8)
    for layer_idx, selections in router_selections.items():
        for expert_idx in range(8):
            expert_counts[expert_idx] += (selections == expert_idx).sum().item()
    total = expert_counts.sum().item()
    return expert_counts, total

print("=" * 60)
print("EXPERT ACTIVATION FREQUENCY BY PROMPT TYPE")
print("=" * 60)

all_results = {}
for prompt_type, prompt in prompts.items():
    counts, total = get_expert_usage(prompt)
    all_results[prompt_type] = counts
    print(f"\n[{prompt_type.upper()}]")
    print(f"Total expert activations: {int(total)}")
    for i, count in enumerate(counts):
        pct = count.item() / total * 100
        bar = "█" * int(pct / 2)
        print(f"  Expert {i}: {bar:<25} {pct:5.1f}%  ({int(count.item())} activations)")

print("\n" + "=" * 60)
print("CROSS-PROMPT EXPERT PREFERENCE")
print("=" * 60)
print(f"\n{'Expert':<10}", end="")
for pt in all_results:
    print(f"{pt:>12}", end="")
print()
print("-" * 58)
for expert_idx in range(8):
    print(f"Expert {expert_idx:<3}", end="")
    for pt, counts in all_results.items():
        total = counts.sum().item()
        pct = counts[expert_idx].item() / total * 100
        print(f"{pct:>11.1f}%", end="")
    print()

print("\n" + "=" * 60)
print("MOST DISTINCTIVE EXPERT PER PROMPT TYPE")
print("=" * 60)
for pt, counts in all_results.items():
    total = counts.sum().item()
    pcts = counts / total * 100
    top_expert = pcts.argmax().item()
    print(f"  {pt:<12}: Expert {top_expert} ({pcts[top_expert]:.1f}%)")

for h in hooks:
    h.remove()
print("\nDone. Hooks removed.")
