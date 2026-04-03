# Infrastructure for Agentic AI
---
[dagc.ai](https://www.dagc.ai) | [LinkedIn](https://www.linkedin.com/in/danielaleguerrero/) | [Part I: AI Infrastructure Learning](https://github.com/dagc-ai/ai-infra-learning)
---

Part I built a mental model from silicon to transformer — CUDA kernels, Ring AllReduce from scratch, vLLM on A100s, quantization benchmarks, and a nanoGPT trained from scratch that traced a single token through every layer of the hardware stack. That repo ends at the model output. This one starts there.

Part II covers everything above the model layer: how modern LLMs are actually architected for agentic workloads, how fine-tuning and alignment shape behavior, how retrieval grounds models in external knowledge, how agents are wired into systems that do real work, and how those systems run reliably in production. Same principle as Part I — every concept gets an exercise, every exercise produces a number, every number gets committed here with the conditions and the honest account of what went wrong.

The throughline in Part I was that the binding constraint in AI systems is almost always data movement, not computation. The throughline in Part II is different: every architectural decision above the model layer exists to solve a reliability problem, not a performance problem. GQA is not faster attention — it is attention that fits in memory at real concurrent load. RAG is not a smarter model — it is a model with a deterministic audit trail for where its answers came from. The agent reliability patterns in Phase 15 are not optional polish — they are the difference between a demo and a deployed system.

The capstone is a fully autonomous content engine: four specialized agents orchestrated by OpenClaw, with CockroachDB as the shared state store, publishing to dagc.ai without human intervention. Every phase contributes a component. The engine is the integration test for the whole curriculum.

---

## Progress

| Phase | Topic | Status | Key Result |
|-------|-------|--------|------------|
| 8 | Modern LLM Architecture for Agentic Workloads | ✅ Complete | GQA reduces KV cache 4x vs. MHA — at 128K context, batch=10: 172GB (GQA) vs. 687GB (MHA). MoE routing measured as uniform across all domain types (cosine similarity > 0.996 between code/math/language/tool-call). |
| 9 | Fine-Tuning: SFT, LoRA, QLoRA | ⬜ Queued | |
| 10 | Alignment: RLHF, DPO, Reward Modeling | ⬜ Queued | |
| 11 | Evals: Measuring Model Behavior | ⬜ Queued | |
| 12 | RAG + Storage Architecture | ⬜ Queued | |
| 13 | Tool Use and the ReAct Pattern | ⬜ Queued | |
| 14 | Agent Frameworks: LangGraph, OpenClaw, CrewAI | ⬜ Queued | |
| 15 | Production Agentic Infrastructure | ⬜ Queued | |
| 16 | Capstone: AI Learning Hub Content Engine | ⬜ Queued | |

---

## Phase 8 — Modern LLM Architecture for Agentic Workloads

**Hardware:** MacBook Pro M5 Max, 128GB unified memory — model inspection and config analysis (no GPU required for weight shape verification). Mixtral 8x7B routing analysis run on A100 80GB via RunPod.
**Models inspected:** Llama 3.1 8B (architecture inspection + KV cache analysis), Mixtral 8x7B Instruct v0.1 (MoE routing analysis)
**Baseline:** nanoGPT GPT-2 style transformer from Phase 7 — every Llama 3.1 divergence is measured against this baseline

### Key Results

**Exercise 1 — Architecture Inspection: nanoGPT vs. Llama 3.1 8B**

All weight shapes pulled from loaded model tensors, not from the paper.

| Feature | nanoGPT (GPT-2) | Llama 3.1 8B | Why It Changed |
|---------|----------------|--------------|----------------|
| Position encoding | Learned table `wpe [1024, 768]` — 786K params, hard boundary at 1,024 tokens | RoPE — 0 learned params, computed as rotation on Q and K at runtime | Agents accumulate long tool call histories; learned tables have a hard wall and cost params proportional to max length |
| Normalization | LayerNorm — weight + bias, subtracts mean | RMSNorm — weight only, no mean subtraction | At 32 layers × 2 norms each, simpler math compounds into measurable throughput gains at inference scale |
| Activation | GELU — smooth nonlinearity | SwiGLU — `gate_proj × SiLU(up_proj)`, then `down_proj` | Gating suppresses irrelevant activations entirely rather than smoothing them; empirically better loss at same parameter count |
| Attention | MHA — Q/K/V all same size, `c_attn [768, 2304]` | GQA — `q_proj [4096, 4096]`, `k_proj [1024, 4096]`, `v_proj [1024, 4096]` | 4:1 Q-to-KV ratio shrinks KV cache 4x — makes 128K context at concurrent agent load feasible on one GPU |
| FFN structure | 2 matrices: expand 4x then contract | 3 matrices: gate + up (both 3.5x) multiplied, then down contracts | SwiGLU requires parallel gate path; 3.5x expansion matches same parameter count as 4x with two matrices |
| Context length | 1,024 tokens | 131,072 tokens | Agentic workflows accumulate tool results across many turns — 1K context is unusable |
| KV heads | 12 (MHA — all heads cache K/V) | 8 GQA heads (32 Q heads share 8 K/V heads) | 4x KV cache reduction |
| Vocab size | 50,257 | 128,256 | Larger vocabulary = fewer tokens per concept = shorter effective sequences for same content |
| Total params | ~124M | 8,030,261,248 | Scale |
| Non-embedding params | ~117M | 7,504,924,672 | The reasoning capacity — embedding table is lookup overhead |
| Memory (bfloat16) | ~0.25GB | 16.1GB | |

Verified weight shapes from loaded tensors:
```
q_proj:    [4096, 4096]   — 32 query heads × 128 head_dim
k_proj:    [1024, 4096]   — 8 KV heads × 128 head_dim  (4x smaller than Q)
v_proj:    [1024, 4096]   — 8 KV heads × 128 head_dim  (4x smaller than Q)
gate_proj: [14336, 4096]
up_proj:   [14336, 4096]
down_proj: [4096, 14336]
RMSNorm:   ['weight']     — no bias parameter
RoPE:      no learned parameters
```

Key numbers to know cold:
- GQA ratio: 32 Q heads, 8 KV heads — 4:1
- RoPE learned parameters: 0
- nanoGPT position table: 786,432 learned parameters, hard wall at 1,024 tokens
- Weights: 16.1GB in bfloat16; fits on one A100 80GB with 63.9GB to spare

**Exercise 2 — KV Cache Arithmetic: MHA vs. GQA at Scale**

Computed programmatically across three model configurations at context lengths of 8K, 32K, and 128K, at batch sizes of 1, 10, and 50.

| Config | Context | Batch=1 | Batch=10 | Batch=50 |
|--------|---------|---------|----------|---------|
| 8B GQA (actual) | 8K | 1.1GB | 10.7GB | 53.7GB |
| 8B GQA (actual) | 32K | 4.3GB | 42.9GB | 214.4GB |
| 8B GQA (actual) | 128K | 17.2GB | 171.8GB | 859.0GB |
| 8B MHA (hypothetical) | 128K | 68.7GB | 687.2GB | 3,436GB |
| 70B GQA | 128K | 42.9GB | 429.5GB | 2,147.5GB |

At 128K context and batch=10: GQA requires 172GB of KV cache. The hypothetical MHA equivalent requires 687GB — more than eight A100 80GBs, just for cache. GQA is the decision that makes 128K context at real concurrency levels physically possible. Without it, long-context agentic serving requires a fleet, not a server.

Capstone implication: the four-agent content engine at 32K context per agent requires 17.2GB of KV cache total, well within a single A100 or the M5 Max unified memory pool. Sequential agent execution means all four caches are never at maximum simultaneously.

**Exercise 3 — MoE Routing Analysis: Mixtral 8x7B**

Two methodologically distinct tests, same result.

*v1:* 1 prompt per domain (code, math, language, tool-call), prefill only, aggregated expert activations across all 32 layers.

*v2:* 5 prompts per domain, 30 generation tokens per prompt, per-layer specialization scores computed, cosine similarity measured between domain routing vectors.

Aggregate expert activation (v2, normalized, uniform baseline = 12.5%):

All experts across all domains activated between 10.9% and 13.6%. Maximum deviation from uniform: 1.6 percentage points.

Per-layer specialization scores:
- Range: 0.000162 to 0.001939
- Meaningful specialization threshold: ~0.01
- Most specialized layer: Layer 7 (0.001939) — still an order of magnitude below meaningful

Domain routing cosine similarity:
```
code vs. math:        0.9992
code vs. language:    0.9973
code vs. tool_call:   0.9994
math vs. language:    0.9988
```
Cosine similarity of 1.0 = identical routing. Every domain pair is above 0.996.

Conclusion: Mixtral routing is effectively uniform across all domain types. The router learns load balancing, not domain specialization.

### What This Means

GPT-2 was built to predict text well. Llama 3.1 was built to run reliably and cheaply inside production systems at scale. Every architectural difference between them is an engineering answer to a real operational problem — not a benchmark optimization, but a deployment constraint. The model that scores highest on a benchmark but costs too much to serve is not a production model.

The MoE finding overturns the common narrative. "Mixture of Experts" implies expert 1 handles code, expert 2 handles math. That is not what is happening. The router distributes load evenly across all experts regardless of input type. MoE's real value is larger total model capacity — more world knowledge in weights — at a fraction of the per-token inference cost. The efficiency gain is real and substantial. The specialization story is a myth.

### Key Insight

Context length is not a free parameter — it is a memory multiplier. Every token in context for every concurrent user requires memory proportional to KV heads × layers × head dimension. At long context and high concurrency, KV cache dominates GPU memory, not model weights. GQA is the architectural decision that keeps this manageable. Understanding this calculation is a qualification tool for infrastructure conversations: "what is your expected context length per session and how many concurrent users do you need to support?" Most buyers have not done this math. The person who runs it in front of them in the first meeting is the person who controls the deal.

---

## Notes

- [Modern LLM Architecture: nanoGPT vs. Llama 3.1 8B](notes/modern-llm-architecture.md) — Phase 8 complete: architecture comparison table (all numbers from loaded model), KV cache arithmetic across MHA vs. GQA vs. MQA, MoE routing analysis v1 and v2

---

## Hardware

| Phase | Hardware | Provider | Cost |
|-------|----------|----------|------|
| 8 | MBP M5 Max 128GB (inspection + config) / A100 SXM4 80GB (Mixtral routing) | Local / RunPod | $1.52/hr |

---

## Companion Repo

Part I — [github.com/dagc-ai/ai-infra-learning](https://github.com/dagc-ai/ai-infra-learning)

Seven phases. Silicon to transformer. CUDA kernels, Ring AllReduce from scratch, vLLM on A100s, quantization benchmarks, Groq vs. A100 head-to-head, and a 30M parameter GPT trained from scratch with deliberate failure modes engineered and documented. The foundation this curriculum builds on.
