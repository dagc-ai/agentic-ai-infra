# Modern LLM Architecture — nanoGPT vs. Llama 3.1 8B
## Phase 8 Deliverable

All weight shapes pulled from loaded model, not from paper.

| Feature | nanoGPT (GPT-2) | Llama 3.1 8B | Why It Changed |
|---------|----------------|--------------|----------------|
| Position encoding | Learned table `wpe [1024, 768]` — 786K params, hard boundary at 1,024 tokens | RoPE — 0 learned params, computed as rotation on Q and K at runtime | Agents need 128K+ context; learned tables have a hard wall and cost parameters proportional to max length |
| Normalization | LayerNorm — weight + bias, subtracts mean | RMSNorm — weight only, no mean subtraction | At 32 layers × 2 norms each, simpler math compounds into measurable throughput gains at inference scale |
| Activation | GELU — smooth nonlinearity | SwiGLU — gate_proj × SiLU(up_proj), then down_proj | Gating suppresses irrelevant activations entirely rather than smoothing them; empirically better loss at same parameter count |
| Attention | MHA — Q/K/V all same size, `c_attn [768, 2304]` | GQA — `q_proj [4096, 4096]`, `k_proj [1024, 4096]`, `v_proj [1024, 4096]` | 4:1 Q-to-KV ratio shrinks KV cache 4x — makes 128K context at concurrent agent load feasible on one GPU |
| FFN structure | 2 matrices: expand 4x then contract | 3 matrices: gate + up (both 3.5x) multiplied, then down contracts | SwiGLU requires parallel gate path; 3.5x expansion hits same parameter count as 4x with two matrices |
| Context length | 1,024 tokens | 131,072 tokens | Agentic workflows accumulate tool results across many turns — 1K context is unusable |
| KV heads | 12 (MHA — all heads cache K/V) | 8 (GQA — 32 Q heads share 8 K/V heads) | See above — 4x KV cache reduction |
| Vocab size | 50,257 | 128,256 | Larger vocabulary = fewer tokens per concept = shorter effective sequences for same content |
| Total params | ~124M | 8,030,261,248 | Scale |
| Non-embedding params | ~117M | 7,504,924,672 | The reasoning capacity — embedding table is lookup overhead |
| Memory (bfloat16) | ~0.25GB | 16.1GB | |

## Key Numbers to Know Cold

- GQA ratio: 32 Q heads, 8 KV heads — 4:1
- KV cache at 128K context, batch=1: 17.2GB
- KV cache at 128K context, batch=10: 171.8GB
- Weights: 16.1GB in bfloat16
- Total fits on one A100 80GB at batch=1 with 46GB to spare
- RoPE learned parameters: 0
- nanoGPT position table: 786,432 learned parameters, hard wall at 1,024

## The One-Sentence Summary

Every architectural change in Llama 3.1 relative to GPT-2 exists to make the model cheaper to serve at scale — more context for less memory, faster normalization, better quality per parameter — because a model that produces good text but costs too much to run is not a production model.
