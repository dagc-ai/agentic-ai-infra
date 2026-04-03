from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

print("Loading Mixtral 8x7B in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Model loaded.\n")

NUM_EXPERTS = 8
NUM_LAYERS  = 32
MAX_NEW_TOKENS = 30  # reduced from 50

prompts = {
    "code": [
        "[INST] Write a Python function that implements binary search. [/INST]",
        "[INST] Write a Python class that implements a stack data structure. [/INST]",
        "[INST] Write a recursive function to compute the nth Fibonacci number. [/INST]",
        "[INST] Write a Python decorator that measures function execution time. [/INST]",
        "[INST] Implement breadth-first search on a graph in Python. [/INST]",
    ],
    "math": [
        "[INST] What is the derivative of x^3 + 2x^2 - 5x + 1? [/INST]",
        "[INST] Solve the quadratic equation 2x^2 + 5x - 3 = 0. [/INST]",
        "[INST] What is the integral of sin(x) * cos(x)? [/INST]",
        "[INST] What is the limit of (sin x)/x as x approaches 0? [/INST]",
        "[INST] Prove that the square root of 2 is irrational. [/INST]",
    ],
    "language": [
        "[INST] Explain the causes of World War I in three sentences. [/INST]",
        "[INST] Summarize the plot of Romeo and Juliet. [/INST]",
        "[INST] What were the main causes of the French Revolution? [/INST]",
        "[INST] Explain the difference between a metaphor and a simile. [/INST]",
        "[INST] What is the significance of the Magna Carta? [/INST]",
    ],
    "tool_call": [
        "[INST] You have access to a weather API. Get the current weather in Austin, Texas and return JSON. [/INST]",
        "[INST] You have access to a search API. Find the current stock price of Apple and return JSON. [/INST]",
        "[INST] You have access to a calendar API. Schedule a meeting for tomorrow at 2pm and return JSON. [/INST]",
        "[INST] You have access to a database API. Query all users created in the last 7 days and return JSON. [/INST]",
        "[INST] You have access to an email API. Send a meeting confirmation to john@example.com and return JSON. [/INST]",
    ],
}

domain_layer_distributions = {domain: [] for domain in prompts}
router_selections = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        router_logits = output[1]
        selected = torch.topk(router_logits, k=2, dim=-1).indices
        if layer_idx not in router_selections:
            router_selections[layer_idx] = []
        router_selections[layer_idx].append(selected.detach().cpu())
    return hook

hooks = []
for layer_idx, layer in enumerate(model.model.layers):
    if hasattr(layer, 'block_sparse_moe'):
        h = layer.block_sparse_moe.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

print(f"Registered hooks on {len(hooks)} MoE layers")
print(f"Running 20 prompts (4 domains x 5 prompts, {MAX_NEW_TOKENS} generation tokens each)\n")

for domain, domain_prompts in prompts.items():
    print(f"Processing domain: {domain.upper()}")
    for i, prompt in enumerate(domain_prompts):
        router_selections.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        layer_dist = np.zeros((NUM_LAYERS, NUM_EXPERTS))
        for layer_idx in range(NUM_LAYERS):
            if layer_idx in router_selections:
                all_selections = torch.cat(router_selections[layer_idx], dim=0)
                for expert_idx in range(NUM_EXPERTS):
                    layer_dist[layer_idx, expert_idx] = (
                        all_selections == expert_idx
                    ).sum().item()
                row_sum = layer_dist[layer_idx].sum()
                if row_sum > 0:
                    layer_dist[layer_idx] /= row_sum

        domain_layer_distributions[domain].append(layer_dist)
        print(f"  Prompt {i+1}/5 done")
    print()

domain_means = {
    domain: np.mean(np.stack(dists), axis=0)
    for domain, dists in domain_layer_distributions.items()
}

print("=" * 70)
print("AGGREGATE EXPERT DISTRIBUTION (mean across 5 prompts, normalized)")
print("Uniform baseline = 12.5% per expert")
print("=" * 70)

for domain, mean_dist in domain_means.items():
    collapsed = mean_dist.mean(axis=0)
    print(f"\n[{domain.upper()}]")
    for i, pct in enumerate(collapsed):
        bar = "█" * int(pct * 200)
        print(f"  Expert {i}: {bar:<25} {pct*100:5.1f}%")

print("\n" + "=" * 70)
print("CROSS-DOMAIN EXPERT PREFERENCE TABLE")
print("=" * 70)
print(f"\n{'Expert':<10}", end="")
for domain in domain_means:
    print(f"{domain:>12}", end="")
print()
print("-" * 58)
for expert_idx in range(NUM_EXPERTS):
    print(f"Expert {expert_idx:<3}", end="")
    for domain, mean_dist in domain_means.items():
        collapsed = mean_dist.mean(axis=0)
        print(f"{collapsed[expert_idx]*100:>11.1f}%", end="")
    print()

print("\n" + "=" * 70)
print("PER-LAYER SPECIALIZATION SCORE")
print("(Variance across domains — higher = more domain-sensitive routing)")
print("=" * 70)
print(f"\n{'Layer':<8} {'Specialization':>15}  {'Most Dominant Expert'}")
print("-" * 50)

layer_specialization = []
for layer_idx in range(NUM_LAYERS):
    layer_by_domain = np.stack([
        domain_means[domain][layer_idx]
        for domain in domain_means
    ])
    spec_score = layer_by_domain.var(axis=0).mean()
    layer_specialization.append(spec_score)
    top_expert = layer_by_domain.mean(axis=0).argmax()
    print(f"Layer {layer_idx:<3} {spec_score:>15.6f}  Expert {top_expert}")

print(f"\nMost specialized layer:  {np.argmax(layer_specialization)} "
      f"(score: {max(layer_specialization):.6f})")
print(f"Least specialized layer: {np.argmin(layer_specialization)} "
      f"(score: {min(layer_specialization):.6f})")
print(f"Mean specialization:     {np.mean(layer_specialization):.6f}")

print("\n" + "=" * 70)
print("DOMAIN ROUTING SIMILARITY (cosine — 1.0 = identical routing)")
print("=" * 70)
domains = list(domain_means.keys())
collapsed = {d: domain_means[d].mean(axis=0) for d in domains}
print(f"\n{'':12}", end="")
for d in domains:
    print(f"{d:>12}", end="")
print()
for d1 in domains:
    print(f"{d1:<12}", end="")
    for d2 in domains:
        v1, v2 = collapsed[d1], collapsed[d2]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"{sim:>12.4f}", end="")
    print()

for h in hooks:
    h.remove()
print("\nDone. Hooks removed.")
