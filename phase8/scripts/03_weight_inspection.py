from transformers import AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.1-8B"

print("Loading weights...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("\n=== FINDING GQA IN THE WEIGHTS ===")
attn = model.model.layers[0].self_attn
print(f"q_proj shape: {attn.q_proj.weight.shape}  — {attn.q_proj.weight.shape[0]} output dims (32 heads × 128 head_dim)")
print(f"k_proj shape: {attn.k_proj.weight.shape}  — {attn.k_proj.weight.shape[0]} output dims (8 heads × 128 head_dim)")
print(f"v_proj shape: {attn.v_proj.weight.shape}  — {attn.v_proj.weight.shape[0]} output dims (8 heads × 128 head_dim)")
print(f"o_proj shape: {attn.o_proj.weight.shape}")
print(f"\nnanoGPT equivalent: c_attn shape would be [768, 2304] — Q+K+V concatenated, all same size")
print(f"Llama: Q is {attn.q_proj.weight.shape[0]}, K is {attn.k_proj.weight.shape[0]} — {attn.q_proj.weight.shape[0] // attn.k_proj.weight.shape[0]}x asymmetry. That's GQA.")

print("\n=== FINDING RMSNorm IN THE WEIGHTS ===")
norm = model.model.layers[0].input_layernorm
print(f"Norm type: {type(norm).__name__}")
param_names = [n for n, _ in norm.named_parameters()]
print(f"Norm parameter names: {param_names}")
print(f"\nnanoGPT LayerNorm has: weight + bias")
print(f"Llama RMSNorm has: weight only — no bias because there's no mean subtraction to re-center")

print("\n=== FINDING SwiGLU IN THE WEIGHTS ===")
mlp = model.model.layers[0].mlp
print(f"gate_proj shape: {mlp.gate_proj.weight.shape}")
print(f"up_proj shape:   {mlp.up_proj.weight.shape}")
print(f"down_proj shape: {mlp.down_proj.weight.shape}")
print(f"\nnanoGPT MLP: c_fc [768, 3072] + c_proj [3072, 768] — two matrices")
print(f"Llama MLP: gate + up + down — three matrices. gate and up run in parallel, multiply together, then down projects back.")
print(f"Intermediate dim: {mlp.gate_proj.weight.shape[0]} ({mlp.gate_proj.weight.shape[0] / 4096:.2f}x hidden dim vs nanoGPT 4.0x)")

print("\n=== FINDING RoPE IN THE WEIGHTS ===")
print("Searching for rotary/rope learned parameters...")
rope_params = [
    (n, p.shape) for n, p in model.named_parameters()
    if "rotary" in n or "rope" in n
]
if rope_params:
    print(f"Found learned RoPE parameters: {rope_params}")
else:
    print("Learned RoPE parameters: NONE")
    print("This IS the finding: RoPE has zero learned parameters.")
    print("nanoGPT wpe was shape [1024, 768] = 786,432 learned parameters just for position.")
    print("Llama position encoding: computed from a formula at runtime. 0 parameters. No hard boundary.")

print(f"\nRoPE implementation type: {type(model.model.layers[0].self_attn.rotary_fn).__name__}")

print("\n=== TOTAL PARAMETER COUNT ===")
total = sum(p.numel() for p in model.parameters())
embed = model.model.embed_tokens.weight.numel()
print(f"Total parameters:      {total:,}")
print(f"Embedding parameters:  {embed:,}  ({embed/total*100:.1f}% of total)")
print(f"Non-embedding params:  {total-embed:,}")
print(f"Memory (bfloat16):     {total * 2 / 1e9:.1f}GB")
