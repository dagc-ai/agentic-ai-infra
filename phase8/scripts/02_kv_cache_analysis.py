def kv_cache_gb(num_layers, num_kv_heads, seq_len, head_dim, batch_size, bytes=2):
    return (2 * num_layers * num_kv_heads * seq_len * head_dim * batch_size * bytes) / 1e9

configs = {
    "Llama 3.1 8B  — GQA (8 KV heads) ": (32, 8,  128),
    "Llama 3.1 8B  — MHA hypothetical  ": (32, 32, 128),
    "Llama 3.1 70B — GQA (8 KV heads) ": (80, 8,  128),
}

seq_lens = [8192, 32768, 131072]
batch_sizes = [1, 10, 50]

for name, (layers, kv_heads, head_dim) in configs.items():
    print(f"\n{name}")
    print(f"{'Seq Len':<12} {'Batch=1':>10} {'Batch=10':>10} {'Batch=50':>10}")
    print("-" * 45)
    for seq in seq_lens:
        row = f"{seq//1024}K tokens   "
        for batch in batch_sizes:
            gb = kv_cache_gb(layers, kv_heads, seq, head_dim, batch)
            row += f"{gb:>9.1f}GB"
        print(row)

print("\n=== A100 80GB Headroom (weights + KV cache) ===")
print("Llama 3.1 8B  weights (bfloat16): ~16GB")
print("Llama 3.1 70B weights (bfloat16): ~140GB")
print("Llama 3.1 70B weights (int4):     ~35GB")
print("\nRemaining for KV cache on one A100 80GB:")
print(f"  8B  bfloat16: {80 - 16:.0f}GB available")
print(f"  70B bfloat16: {80 - 140:.0f}GB available (negative — needs multiple GPUs)")
print(f"  70B int4:     {80 - 35:.0f}GB available")
