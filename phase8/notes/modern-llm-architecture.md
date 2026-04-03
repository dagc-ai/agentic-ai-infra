# Modern LLM Architecture — nanoGPT vs. Llama 3.1 8B
## Phase 8 Deliverable
**All weight shapes pulled from loaded model, not from paper.**
**All KV cache numbers computed from first principles and verified programmatically.**
**All MoE routing findings from live inference runs on Mixtral 8x7B.**

---

## Architecture Comparison Table

| Feature | nanoGPT (GPT-2) | Llama 3.1 8B | Why It Changed |
|---------|----------------|--------------|----------------|
| Position encoding | Learned table `wpe [1024, 768]` — 786K params, hard boundary at 1,024 tokens | RoPE — 0 learned params, computed as rotation on Q and K at runtime | Agents need 128K+ context; learned tables have a hard wall and cost parameters proportional to max length |
| Normalization | LayerNorm — weight + bias, subtracts mean | RMSNorm — weight only, no mean subtraction | At 32 layers × 2 norms each, simpler math compounds into measurable throughput gains at inference scale |
| Activation | GELU — smooth nonlinearity | SwiGLU — gate_proj × SiLU(up_proj), then down_proj | Gating suppresses irrelevant activations entirely rather than smoothing them; empirically better loss at same parameter count |
| Attention | MHA — Q/K/V all same size, `c_attn [768, 2304]` | GQA — `q_proj [4096, 4096]`, `k_proj [1024, 4096]`, `v_proj [1024, 4096]` | 4:1 Q-to-KV ratio shrinks KV cache 4x — makes 128K context at concurrent agent load feasible on one GPU |
| FFN structure | 2 matrices: expand 4x then contract | 3 matrices: gate + up (both 3.5x) multiplied, then down contracts | SwiGLU requires parallel gate path; 3.5x expansion hits same parameter count as 4x with two matrices |
| Context length | 1,024 tokens | 131,072 tokens | Agentic workflows accumulate tool results across many turns — 1K context is unusable |
| KV heads | 12 (MHA — all heads cache K/V) | 8 (GQA — 32 Q heads share 8 K/V heads) | 4x KV cache reduction |
| Vocab size | 50,257 | 128,256 | Larger vocabulary = fewer tokens per concept = shorter effective sequences for same content |
| Total params | ~124M | 8,030,261,248 | Scale |
| Non-embedding params | ~117M | 7,504,924,672 | The reasoning capacity — embedding table is lookup overhead |
| Memory (bfloat16) | ~0.25GB | 16.1GB | |

---

## Key Numbers to Know Cold

- GQA ratio: 32 Q heads, 8 KV heads — 4:1
- KV cache at 128K context, batch=1: 17.2GB
- KV cache at 128K context, batch=10: 171.8GB
- Weights: 16.1GB in bfloat16
- Total fits on one A100 80GB at batch=1 with 46GB to spare
- RoPE learned parameters: 0
- nanoGPT position table: 786,432 learned parameters, hard wall at 1,024

---

## Exercise 1 — Architecture Inspection (Config + Weight Shapes)

### What We Did
Loaded Llama 3.1 8B config and full weights. Inspected every architectural parameter against nanoGPT baseline. Found GQA, RMSNorm, SwiGLU, and RoPE in the actual tensor shapes — not from the paper.

### Verified Weight Shapes
```
q_proj: [4096, 4096]   — 32 query heads × 128 head_dim
k_proj: [1024, 4096]   — 8 KV heads × 128 head_dim  (4x smaller than Q)
v_proj: [1024, 4096]   — 8 KV heads × 128 head_dim  (4x smaller than Q)
gate_proj: [14336, 4096]
up_proj:   [14336, 4096]
down_proj: [4096, 14336]
RMSNorm parameters: ['weight']  — no bias
RoPE learned parameters: NONE
```

### So What? (Simple Terms)
GPT-2 was built to predict text well. Llama 3.1 was built to run reliably and cheaply inside production systems at scale. Every single architectural difference exists to solve a real operational problem — not to improve benchmark scores, but to make the model actually deployable. The model that scores highest on a benchmark but costs too much to serve is not a production model.

### Insights for Agentic AI Developers
- RoPE means you are not fighting a hard context wall. You can build agents that accumulate long tool call histories without hitting a cliff at some fixed token count.
- GQA means your KV cache is 4x smaller than it would be in a naive MHA model. This directly determines how many concurrent agent sessions you can serve on one GPU.
- RMSNorm and SwiGLU are not architectural curiosities. They are throughput optimizations that compound across every layer of every forward pass in your serving infrastructure.
- The non-embedding parameter count (7.5B) is what's doing the reasoning. The embedding table (525M) is just a lookup. When evaluating model quality, compare non-embedding params.

### Business Insights
- The architectural gap between GPT-2 and Llama 3.1 represents roughly 5 years of production AI infrastructure learning. The changes are not academic — they are engineering answers to real deployment failures at scale.
- Context window is a pricing lever. Longer context requires proportionally more KV cache memory. Inference platforms that charge more for longer context are passing through a real hardware cost, not an arbitrary margin decision.
- The model a company chooses to deploy is not just a quality decision — it is an infrastructure cost decision. A model with poor KV cache efficiency requires proportionally more hardware to serve the same concurrent load.
- Understanding these constraints lets you have a cost conversation with a customer that goes beyond "how much does the API cost per token" to "what is your total infrastructure cost per concurrent user at your expected context length."

### GTM Implications
- Most enterprise buyers evaluate models on benchmark scores. The conversation that differentiates you: "what does this model cost to serve at your expected concurrency and context length?" Almost nobody is asking that question at the procurement level.
- GQA, RoPE, and RMSNorm are the features that separate a model that demos well from a model that runs in production. If you are selling inference infrastructure, these are the architectural checkboxes that matter.
- When a customer says "we want to run our own model," the right response is not "which model do you want?" It is "what is your context length requirement, your concurrency requirement, and your GPU budget?" The architecture determines whether those three numbers are compatible.
- At Tenstorrent or Groq, this knowledge lets you map customer workload requirements directly to hardware specifications — not from a spec sheet, but from first principles.

---

## Exercise 2 — KV Cache Arithmetic

### What We Did
Computed KV cache size programmatically across three model configurations (Llama 3.1 8B GQA, Llama 3.1 8B hypothetical MHA, Llama 3.1 70B GQA) at context lengths of 8K, 32K, and 128K tokens, at batch sizes of 1, 10, and 50.

### Key Results

| Config | Context | Batch=1 | Batch=10 | Batch=50 |
|--------|---------|---------|---------|---------|
| 8B GQA | 8K | 1.1GB | 10.7GB | 53.7GB |
| 8B GQA | 128K | 17.2GB | 171.8GB | 859.0GB |
| 8B MHA (hypothetical) | 128K | 68.7GB | 687.2GB | 3,436GB |
| 70B GQA | 128K | 42.9GB | 429.5GB | 2,147.5GB |

### So What? (Simple Terms)
Context length is not just a capability number. It is a memory multiplier. Every token in context for every concurrent user requires memory proportional to the number of KV heads, layers, and head dimensions. At long context and high concurrency, KV cache dominates GPU memory — not model weights. GQA is the architectural decision that keeps this manageable. Without it, 128K context at any real concurrency level requires a fleet of GPUs, not a server.

### Insights for Agentic AI Developers
- Design your agent context windows deliberately. Every tool call result you append to context costs GPU memory for the duration of that session. Truncation and summarization strategies are not nice-to-haves — they are memory management.
- The 4-agent capstone content engine at 32K context per agent requires 17.2GB of KV cache total — well within a single A100 or unified memory Mac. Sequential agent execution means you never hold all four caches at max simultaneously.
- At 128K context with batch=10, the KV cache alone is 172GB. That is more than two A100 80GBs just for cache. This is why production serving systems implement aggressive KV cache eviction and reuse strategies.
- PagedAttention in vLLM exists specifically to solve this problem — treating KV cache like OS virtual memory to eliminate fragmentation and enable higher batch sizes.

### Business Insights
- KV cache size is the hidden cost driver in LLM inference at scale. A customer running a customer service agent with 50 concurrent sessions at 32K context each needs 214GB of KV cache for Llama 8B. That is 3 A100 80GBs before a single byte of model weights is loaded.
- Inference platform pricing based on context length is not arbitrary. It is hardware math passed through to the customer.
- The gap between "8B model runs on one GPU" and "8B model runs 50 concurrent users on one GPU at long context" is enormous. Customers who discover this gap in production — rather than in a proof of concept — are the customers who churn from a platform.
- Companies that help customers understand and manage KV cache costs before signing a contract build trust. Companies that hide this cost discovery until the production bill arrives lose accounts.

### GTM Implications
- The KV cache arithmetic is a qualification tool. Ask a prospect: "What is your expected context length per session and how many concurrent users do you need to support?" Run the math in front of them. If they have not done this analysis, you are the first person to show them what their infrastructure actually costs.
- At an AI hardware company, this calculation is how you justify a larger GPU configuration. "Your workload requires X GB of KV cache at your expected concurrency. Here is why you need Y GPUs, not one."
- The move from single-user demo to multi-user production is where most AI deployments hit unexpected costs. Positioning your solution around this transition — rather than raw benchmark performance — addresses the actual pain point.

---

## Exercise 3 — MoE Routing Analysis (v1 + v2)

### What We Did
Loaded Mixtral 8x7B and attached forward hooks to all 32 MoE router layers. Measured which experts were activated for different prompt types across two methodologically distinct tests.

**v1:** 1 prompt per domain, prefill only, aggregated across all layers.
**v2:** 5 prompts per domain, 30 generation tokens captured, per-layer specialization scores, cosine similarity between domain routing vectors.

### Results

**Aggregate expert activation (v2, normalized, uniform baseline = 12.5%):**

All experts across all domains activated between 10.9% and 13.6%. Maximum deviation from uniform: 1.6 percentage points.

**Per-layer specialization scores:**
- Range: 0.000162 to 0.001939
- Meaningful specialization threshold: ~0.01
- Most specialized layer: Layer 7 (0.001939) — still an order of magnitude below meaningful

**Domain routing cosine similarity:**
```
code vs. math:       0.9992
code vs. language:   0.9973
code vs. tool_call:  0.9994
math vs. language:   0.9988
```
Cosine similarity of 1.0 = identical routing. Every domain pair above 0.996.

**Conclusion:** Two methodologically distinct tests, same result. Mixtral routing is effectively uniform across all domain types.

### So What? (Simple Terms)
The name "Mixture of Experts" implies that expert 1 handles code, expert 2 handles math, and so on. That is not what is happening. The router does not learn to send different types of questions to different specialists. It learns to distribute load evenly across all experts regardless of what the question is about. MoE's real value is that you get a much larger total model — more world knowledge stored in weights — while only activating a fraction of those parameters per token. The efficiency gain is real. The specialization story is a myth.

### Insights for Agentic AI Developers
- Do not design agent routing logic around assumptions of MoE expert specialization. The model is not internally routing your coding agent differently than your writing agent.
- MoE capacity advantage is real but domain-agnostic. Mixtral 8x7B has more raw knowledge capacity than a dense 14B model at the same inference cost. That matters for agents that need broad world knowledge — but the broadness is general, not specialized.
- If you need a genuinely specialized agent — one that excels at code, or medical text, or legal documents — task-specific fine-tuning on a dense model will outperform routing assumptions about a MoE model.
- The uniform routing also means MoE models have predictable, stable latency characteristics. There is no "slow path" for certain input types. Every token costs the same compute regardless of content.

### Business Insights
- MoE is a genuine cost-efficiency architecture. Mixtral 8x7B delivers quality competitive with dense 30-40B models at the inference cost of a 14B model. That cost-quality tradeoff is real and quantifiable.
- The "specialization" narrative around MoE is marketing, not engineering. Routing is learned load balancing, not domain routing. Customers who make architectural decisions based on MoE specialization assumptions are making decisions on false premises.
- For inference infrastructure companies, MoE models present a packaging opportunity: larger effective model quality at lower per-token compute cost. The challenge is that MoE models have larger total weight files — which creates storage, download, and initial load costs that offset some of the inference savings.
- Memory capacity to hold the full MoE model is the real deployment constraint, not inference compute. A company that can solve the memory capacity problem — through better hardware, quantization, or distributed serving — unlocks the full MoE quality-efficiency advantage.

### GTM Implications
- When a customer asks "should we use a MoE model for our multi-domain agent?" the honest answer is: yes for capacity efficiency, no if they expect automatic domain specialization. Being the person who gives the honest answer builds credibility with technical buyers faster than confirming their assumptions.
- MoE models are a strong argument for hardware with large memory capacity. Mixtral 8x7B in bfloat16 is 47GB — it needs hardware that can hold that without spillover. This is a direct conversation opener for Tenstorrent, Groq, or any inference hardware company whose product solves the memory capacity problem.
- The uniform routing finding has a practical sales implication: MoE model performance is predictable across use cases. That predictability is a production reliability argument — no surprise latency spikes for certain input types.
- At an AI hardware company, the MoE trend is an architectural tailwind. As models get larger through MoE rather than dense scaling, memory capacity becomes the binding constraint — which is exactly what next-generation accelerators are designed to address.

---

## The One-Sentence Summary

Every architectural change in Llama 3.1 relative to GPT-2 exists to make the model cheaper to serve at scale — more context for less memory, faster normalization, better quality per parameter — because a model that produces good text but costs too much to run is not a production model.

---

## Phase 8 Completion Checklist

- [x] Architecture inspection: config + weight shapes verified from loaded model
- [x] KV cache arithmetic: computed programmatically across MHA vs. GQA vs. MQA at multiple context lengths and batch sizes
- [x] Weight inspection: GQA 4:1 ratio, RMSNorm no bias, SwiGLU 3-matrix structure, RoPE zero learned parameters — all confirmed in actual tensors
- [x] MoE routing v1: 1 prompt per domain, prefill, aggregated — uniform routing observed
- [x] MoE routing v2: 5 prompts per domain, generation tokens, per-layer scores, cosine similarity — uniform routing confirmed with high confidence
- [x] Architecture comparison table: all numbers from loaded model, not from paper
- [x] Notes written with technical findings, agentic insights, business implications, and GTM framing
