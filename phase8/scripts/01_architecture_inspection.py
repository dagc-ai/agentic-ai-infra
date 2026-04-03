from transformers import AutoConfig
import torch

model_id = "meta-llama/Llama-3.1-8B"

print("Downloading config (no weights)...")
config = AutoConfig.from_pretrained(model_id)

print("\n=== Llama 3.1 8B Config ===")
print(f"Hidden dim (d_model):       {config.hidden_size}")
print(f"Num attention heads:         {config.num_attention_heads}")
print(f"Num KV heads (GQA):          {config.num_key_value_heads}")
print(f"Head dim:                    {config.hidden_size // config.num_attention_heads}")
print(f"Num layers:                  {config.num_hidden_layers}")
print(f"FFN intermediate dim:        {config.intermediate_size}")
print(f"Max position embeddings:     {config.max_position_embeddings}")
print(f"RoPE theta:                  {config.rope_scaling}")
print(f"Vocab size:                  {config.vocab_size}")
print(f"Hidden act:                  {config.hidden_act}")
print(f"Tie word embeddings:         {config.tie_word_embeddings}")

print("\n=== nanoGPT (GPT-2 small) Baseline ===")
print(f"Hidden dim (d_model):        768")
print(f"Num attention heads:         12")
print(f"Num KV heads (MHA):          12  # every head has its own K/V")
print(f"Head dim:                    64")
print(f"Num layers:                  12")
print(f"FFN intermediate dim:        3072  # exactly 4x d_model")
print(f"Max position embeddings:     1024  # hard wall")
print(f"RoPE theta:                  N/A   # learned absolute position table (wpe)")
print(f"Vocab size:                  50257")
print(f"Hidden act:                  gelu")

print("\n=== Key Ratios ===")
q_heads = config.num_attention_heads
kv_heads = config.num_key_value_heads
print(f"Q heads : KV heads ratio:    {q_heads} : {kv_heads}  ({q_heads // kv_heads}x more Q than KV)")
print(f"Context window ratio:        {config.max_position_embeddings} : 1024  ({config.max_position_embeddings // 1024}x larger)")
print(f"FFN expansion ratio:         {config.intermediate_size / config.hidden_size:.2f}x  (nanoGPT: 4.0x)")
print(f"Depth ratio:                 {config.num_hidden_layers} : 12  ({config.num_hidden_layers // 12}x deeper)")
