# Pt II: Infrastructure for Agentic AI
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
| 9 | Fine-Tuning: SFT, LoRA, QLoRA | ✅ Complete | QLoRA fine-tuned Llama 3.1 8B on 550 domain Q&A pairs in 5m10s on one A100 80GB, adapter 161MB (0.52% of params), final loss 1.533. Rank experiment r=4 to r=64: loss improves continuously (1.81 to 1.20), training time rank-invariant at ~310s. Fine-tuning corrected concrete hallucinations; introduced new failure mode from Socratic training data format. |
| 10 | Alignment: RLHF, DPO, Reward Modeling | ⬜ Queued | Reward model trained on preference pairs, PPO-based RLHF mechanics, DPO loss function and why it displaced PPO in practice, SFT vs. DPO qualitative comparison on 20 test prompts |
| 11 | Evals: Measuring Model Behavior | ⬜ Queued | LLM-as-judge eval harness built from scratch, 50-prompt domain-specific test set, calibrated rubric for AI infrastructure content quality — becomes the Editor agent's decision function in Phase 16 |
| 12 | RAG + Storage Architecture | ⬜ Queued | RAG pipeline built from scratch (no framework), three-way storage benchmark: Chroma vs. pgvector vs. CockroachDB under 4-agent concurrent write load, HyDE and reranking implemented and measured against baseline |
| 13 | Tool Use and the ReAct Pattern | ⬜ Queued | Raw function calling with the Anthropic API, explicit ReAct thought/action/observation loop, stateful 5-step research agent, failure modes documented when tools fail or the model loops |
| 14 | Agent Frameworks: LangGraph, OpenClaw, CrewAI | ⬜ Queued | LangGraph research agent with checkpointing, OpenClaw source dissection and custom content engine skill, CrewAI multi-agent content crew dry run, framework tradeoff comparison |
| 15 | Production Agentic Infrastructure | ⬜ Queued | Full agent tracing in LangSmith/Langfuse, cost model per content engine run, retry/fallback/circuit breaker patterns, concurrent agent state load test — no lost updates under 4-agent write contention |
| 16 | Capstone: AI Learning Hub Content Engine | ⬜ Queued | Four agents (Researcher, Writer, Editor, Publisher) orchestrated by OpenClaw, CockroachDB + pgvector as shared state store, fully autonomous post to dagc.ai from a single Telegram message |

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
 
## Phase 9 — Fine-Tuning: SFT, LoRA, QLoRA
 
**Hardware:** A100 SXM4 80GB, CUDA 12.4, PyTorch 2.4.0, transformers 4.44.0, trl 0.9.6, bitsandbytes 0.46.1, RunPod
**Model:** Llama 3.1 8B Instruct (base), fine-tuned on 550 AI infrastructure Q&A pairs generated from Phases 1-6 curriculum
**Stack note:** bitsandbytes version must be pinned precisely — mismatches produce cryptic runtime errors, not clear warnings. Pin the full stack before running.
 
### Key Results
 
**Exercise 1 — Dataset Construction**
 
589 raw Q&A pairs extracted from 6 curriculum threads (Phases 1-6), filtered and validated to 550 training pairs via a reproducible build pipeline. Failure modes encountered and solved: JSON encoding failures from unescaped quotes inside code examples, off-topic pairs from setup discussions, meta-references that broke pair self-containment. Minimum response length enforced at 75 words to filter shallow Q&A that adds noise without signal.
 
| Metric | Value |
|--------|-------|
| Raw pairs extracted | 589 |
| Pairs after filtering | 550 |
| Filter rate | 6.6% |
| Source threads | 6 (Phases 1-6) |
| Min response length enforced | 75 words |
 
The data preparation pipeline is the same pipeline production teams run at 100,000 pairs — ingestion, generation, quality filtering, deduplication, versioning. The difference is orchestration and scale, not concept.
 
**Exercise 2 — QLoRA Fine-Tuning End to End**
 
Fine-tuned Llama 3.1 8B Instruct using QLoRA (NF4 base + BF16 adapters) at rank=16, targeting all attention projection and FFN layers. Training ran on a single A100 80GB.
 
| Metric | Value |
|--------|-------|
| Training time | 5m 10s |
| Epochs | 3 |
| Steps | 102 |
| Loss at step 1 | 2.51 |
| Final training loss | 1.533 |
| Adapter size | 161MB |
| Trainable parameters | 41,943,040 (0.52% of total) |
| Base model VRAM (NF4) | 19GB |
| Total VRAM utilization | ~23% of 80GB |
 
The base model consumes 19GB in NF4. The adapter and optimizer state add ~2GB. 80GB A100 at 23% utilization — enough headroom to 4x the dataset size or move to a 70B base without changing hardware. Adapter saved at 161MB against a 16.1GB base model: one base model in VRAM, many adapters hot-swapped at request time is the production serving pattern.
 
**Exercise 3 — Rank Sensitivity Experiment**
 
Five adapters trained at r=4 through r=64 on identical data with identical hyperparameters. Key finding: loss improves continuously with rank on this dataset, training time is rank-invariant.
 
| Rank | Trainable Params | Final Loss | Training Time |
|------|-----------------|------------|---------------|
| r=4  | 10,485,760 | 1.8055 | 312s |
| r=8  | 20,971,520 | 1.6771 | 308s |
| r=16 | 41,943,040 | 1.5330 | 310s |
| r=32 | 83,886,080 | 1.3784 | 309s |
| r=64 | 167,772,160 | 1.1982 | 312s |
 
The expected plateau at r=16 did not materialize. Technical AI infrastructure content — specific numbers, reasoning patterns, hardware vocabulary — has higher intrinsic dimensionality than simple instruction-following tasks. Practical sweet spot for this dataset: r=32, best loss-to-parameter tradeoff before adapter size doubles again with marginal return. Training time is rank-invariant because adapter parameters are negligible relative to the frozen base model — doubling rank costs nothing in wall clock time, only in adapter file size.
 
**Exercise 4 — Qualitative Before/After Comparison**
 
10 AI infrastructure prompts run against base Llama 3.1 8B and against the r=32 fine-tuned adapter. Results documented in before-after-comparison.md.
 
Concrete hallucinations corrected by fine-tuning:
- Base model described the roofline model as a psychology framework by Daniel Kahneman. Fine-tuned model correctly described it as a GPU performance analysis tool with arithmetic intensity on one axis and compute/bandwidth bounds on the other.
- Base model described Tenstorrent as a Chinese chip designer. Fine-tuned model gave a conceptually accurate answer about the SRAM-centric architectural bet.
 
Fine-tuning also introduced a new failure mode: the fine-tuned model generates follow-up questions instead of answers on a subset of prompts. Root cause: some training pairs used a Socratic format. The model learned to reproduce the format, not just the content. A training data artifact, not a model failure.
 
Topics with thin training data coverage (Chinchilla scaling laws) remained weak — fine-tuning does not conjure knowledge that was not in the training data. For factual grounding on new material, RAG is the right tool (Phase 12).
 
### What This Means
 
SFT, LoRA, and QLoRA are not alternatives — they are layers that stack. SFT is the training objective. LoRA is the parameter efficiency technique applied on top. QLoRA adds 4-bit quantization of the frozen base weights on top of that. Every QLoRA run is also a LoRA run and also an SFT run. Conflating them is the sign of someone who has read about fine-tuning without running it.
 
The rank experiment overturned the default assumption. r=16 is reasonable for simple tasks like format compliance or persona adoption. For technically dense domains where the behavioral target has high intrinsic dimensionality, it undershoots. Run the experiment — the cost is the same regardless of rank.
 
### Key Insight
 
The data pipeline is the actual competitive moat in enterprise fine-tuning. A single A100 80GB at $1.49/hr running a QLoRA job that costs under $0.15 and completes in five minutes is not a differentiator — it is table stakes. The curated, domain-specific, high-quality training dataset that took months and domain expertise to build is what competitors cannot replicate. A customer with 10 years of support tickets, analyst reports, or expert internal documentation has latent training signal that no foundation model provider can match. The question that wins the enterprise fine-tuning conversation is not "which model?" — it is "what does your data pipeline look like?"
 
---
 
## Notes
 
- [Modern LLM Architecture: nanoGPT vs. Llama 3.1 8B](notes/modern-llm-architecture.md) — Phase 8 complete: architecture comparison table (all numbers from loaded model), KV cache arithmetic across MHA vs. GQA vs. MQA, MoE routing analysis v1 and v2
- [Fine-Tuning Mental Model: SFT, LoRA, QLoRA](notes/fine-tuning-mental-model.md) — Phase 9 complete: how SFT/LoRA/QLoRA stack as layers not alternatives, QLoRA end-to-end on Llama 3.1 8B, rank sensitivity experiment r=4 to r=64 (no plateau on technical domain data), qualitative hallucination correction and new failure mode from Socratic training format
- [Alignment Techniques: RLHF, DPO, Reward Modeling](notes/alignment-techniques.md) — Phase 10: the alignment gap, reward model training on preference pairs, PPO-based RLHF mechanics and why it's unstable at scale, DPO loss function and why it displaced PPO, Constitutional AI
- [Evals: Measuring Model Behavior](notes/evals-mental-model.md) — Phase 11: why benchmarks are unreliable proxies for task performance, the contamination problem, LLM-as-judge methodology and calibration, building domain-specific eval rubrics, RAGAS for RAG pipelines
- [Storage Architecture for Agentic AI](notes/storage-architecture-decision.md) — Phase 12: embeddings from first principles, four types of agent memory and the right storage backend for each, why dedicated vector databases are the wrong default for production agents, pgvector hybrid queries, CockroachDB consistency guarantees under concurrent agent write load
- [Tool Use and the ReAct Pattern](notes/react-pattern.md) — Phase 13: raw function calling with the Anthropic API, the ReAct thought/action/observation loop, explicit state management across multi-turn agent workflows, failure modes when tools fail or the model loops
- [Agent Framework Comparison: LangGraph, OpenClaw, CrewAI](notes/agent-framework-comparison.md) — Phase 14: agents as state machines (LangGraph), OpenClaw architecture dissection (persistent memory, skills system, context management), CrewAI multi-agent orchestration, when to choose each framework
- [Production Agentic Infrastructure](notes/production-agentic-infra.md) — Phase 15: full agent tracing with LangSmith/Langfuse, cost modeling per agent run, retry/fallback/circuit breaker patterns, concurrent agent state under contention, prompt injection defense
- [Content Engine Architecture](notes/content-engine-architecture.md) — Phase 16 capstone: every component, every binding constraint, every agent handoff — the synthesis document for Part II
 
---
 
## Hardware
 
| Phase | Hardware | Provider | Cost |
|-------|----------|----------|------|
| 8 | MBP M5 Max 128GB (inspection + config) / A100 SXM4 80GB (Mixtral routing) | Local / RunPod | — |
| 9 | A100 SXM4 80GB | RunPod | ~$0.15 (5m10s training run) |
 
---
 
## Companion Repo
 
Part I — [github.com/dagc-ai/ai-infra-learning](https://github.com/dagc-ai/ai-infra-learning)
 
Seven phases. Silicon to transformer. CUDA kernels, Ring AllReduce from scratch, vLLM on A100s, quantization benchmarks, Groq vs. A100 head-to-head, and a 30M parameter GPT trained from scratch with deliberate failure modes engineered and documented. The foundation this curriculum builds on.
