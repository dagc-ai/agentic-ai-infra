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
| 10 | Alignment: RLHF, DPO, Reward Modeling | ✅ Complete | Reward model: 0.75 accuracy, 1.46 margin on HH-RLHF (5K pairs, Llama 3.2 1B base). DPO: 0.58 accuracy, 0.44 margin on 1948 pairs. LLM-as-judge counterintuitive finding: BASE scored highest (2.30) — SFT amplified confident hallucination (1.93), DPO marginal recovery (1.83). SFT+DPO scored lower than base; rubric design explains the contradiction. |
| 11 | Evals: Measuring Model Behavior | ✅ Complete | 50-prompt task-specific eval set, 7 categories, 150 total responses scored. Calibration: judge vs. human r=0.861 (passes 0.75 threshold). Contamination test: 0/10 signals — hallucination is confabulation, not memorization. Clean two-by-two: fine-tuning improved style dimensions (mechanistic depth +0.72, audience calibration +0.48), degraded accuracy dimensions (technical accuracy -0.20, calibration -0.42). No variant averaged above 2.6/5 — Editor gate is required, not optional. |
| 12 | RAG + Storage Architecture | ✅ Complete | MPNet selected over MiniLM on paraphrase bridging (+0.143). RAG from scratch: 85% hit rate, 80% top-1. HyDE: +15% hit rate, +10% top-1. Reranker rejected — domain mismatch cost 20 points of top-1 accuracy. RAGAS surfaced the critical failure: Query 7 hit rate 100%, answer relevancy 0.000 — chunking failure invisible to retrieval metrics. Storage benchmark: Chroma 1,258 doc/s but wrong choice; CockroachDB 1.67x concurrent write speedup (383 → 640 doc/s); pgvector/CRDB p99 tighter than Postgres despite no vector index (4.3ms vs 7.7ms). |
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

## Phase 10 — Alignment: RLHF, DPO, Reward Modeling

**Hardware:** A100 SXM4 80GB, CUDA 12.4.1, PyTorch 2.4.0+cu121, transformers 4.44.0, trl 0.9.6, bitsandbytes 0.46.1, RunPod
**Models:** Llama 3.2 1B Instruct (reward model base), Llama 3.1 8B Instruct + Phase 9 SFT adapter (DPO target)
**Stack note:** Full stack downgrade required from Phase 9 verified configuration. Transformers 5.x breaks 4-bit quantization on PyTorch 2.4. accelerate 1.x breaks NF4 quantization dispatch. Pin everything before running.

### Key Results

**Exercise 10.1 — Reward Model Training**

Trained on 5,000 preference pairs from Anthropic HH-RLHF (160,800 total available). Bradley-Terry loss: `L = -log sigmoid(r_chosen - r_rejected)`. Base model replaced language model head with a randomly initialized scalar head — the `score.weight MISSING` warning at load is expected, not an error.

| Metric | Value |
|--------|-------|
| Base model | Llama 3.2 1B Instruct |
| Dataset | HH-RLHF, 5,000 pairs |
| Final loss | 0.6445 |
| Final accuracy | 0.75 |
| Final reward margin | 1.4609 |
| Training time | ~1 hour |
| Estimated cost | ~$2 |

Accuracy of 0.75 means the reward model correctly identifies the human-preferred response in 3 of 4 pairs (random baseline: 0.50). The reward margin of 1.46 is the more meaningful metric — it measures how confidently the model discriminates, not just whether it gets the direction right. Individual step accuracy bounced between 0.25 and 1.0 throughout training at batch size 4; the trend over many steps is what matters.

**Exercise 10.2 — DPO Training**

Starting point: Llama 3.1 8B in 4-bit NF4 with Phase 9 LoRA adapter (r=16). Dataset: 1,948 HH-RLHF preference pairs reformatted to (prompt, chosen, rejected) triples. `ref_model=None` in TRL 0.9.6 derives reference behavior by disabling/enabling the PEFT adapter rather than loading a second model copy — halves the memory requirement.

| Metric | Value |
|--------|-------|
| Beta | 0.1 |
| Learning rate | 5e-7 |
| Dataset | HH-RLHF, 1,948 pairs (after filtering) |
| Final train loss | 0.90 |
| Final accuracy | 0.58 |
| Final reward margin | 0.44 |
| VRAM usage | 51GB / 80GB (64%) |
| Training time | ~16 minutes |
| Estimated cost | ~$0.50 |

Critical hyperparameter: DPO learning rate must be ~5e-7, not SFT-scale 2e-4. SFT-scale learning rates overwrite base model capabilities rather than nudging behavioral tendencies. This is the most important difference between SFT and DPO training configuration.

**Exercise 10.3 — Qualitative Three-Way Comparison**

10 AI infrastructure prompts run against BASE, SFT-only (Phase 9), and SFT+DPO. Each variant loaded fresh from base to prevent adapter stacking.

| Prompt Topic | BASE | SFT | DPO | Notes |
|---|---|---|---|---|
| GQA definition | Wrong (visual QA benchmark) | Wrong (visual QA benchmark) | Wrong (visual QA benchmark) | Training data gap — all three hallucinate identically |
| Flash Attention | Wrong (sparse attention) | Correct O(N²) to O(N) | Correct, minor errors | SFT major improvement |
| Ring AllReduce | Cookie analogy, incomplete | Correct two-phase algorithm | Correct + tree vs. ring tradeoff | SFT major improvement, DPO additive |
| KV cache scaling | Generic DB caching | Correct, wrong tensor shape | Correct shape, cleaner | SFT major improvement |
| DPO definition | Wrong (Differential Privacy) | Wrong (Data Poisoning) | Wrong (Data Poisoning) | Training data gap — all three hallucinate |
| LoRA vs fine-tuning | Wrong acronym, vague | Correct mechanism, fabricated benchmarks | Correct, fewer fabrications | SFT clear improvement |

SFT improvement is dramatic on domain-covered topics. DPO improvement is marginal and specific — 2,000 general helpfulness preference pairs applied to a domain-specific SFT adapter produces weak alignment signal. Topics absent from the 550 SFT pairs hallucinate identically across all three variants. Alignment cannot fix what fine-tuning did not teach.

**Exercise 10.4 — LLM-as-Judge Scoring**

Claude Sonnet scored all three variants on 10 prompts across three dimensions (technical accuracy, conciseness, hallucination avoidance).

| Model | Technical Accuracy | Conciseness | Hallucination Avoidance | Mean |
|-------|-------------------|-------------|------------------------|------|
| BASE | 2.20 | 2.50 | 2.20 | 2.30 |
| SFT | 2.40 | 2.20 | 1.20 | 1.93 |
| DPO | 2.00 | 2.20 | 1.30 | 1.83 |

The counterintuitive result: BASE scored highest overall. This is not evidence that training degraded quality — it reflects what each model was optimized for. Hallucination avoidance tells the real story: SFT taught the model to produce confident, specific responses, including when it had to fabricate specifics. DPO partially recovered hallucination avoidance (1.30 vs. 1.20) but the signal was weak. The judge correctly caught the GQA and DPO acronym hallucinations across all three variants, validating it as a reliable quality gate.

### What This Means

SFT teaches a style, not just knowledge. The model learned to produce confident, specific, technical prose with numbers. That style is valuable when the model knows the answer. It is dangerous when it does not — hallucinations now look identical to correct answers. DPO nudges but does not fix; it addressed neither knowledge gaps nor the confident hallucination amplification introduced by SFT. The binding constraint at the alignment layer is training data quality and coverage, not the sophistication of the alignment algorithm. You cannot DPO your way to domain expertise.

### Key Insight

Failure modes documented alongside successes are first-class deliverables. The GQA and DPO hallucinations that appeared identically across all three model variants are not a sign of a failed phase — they are a precise diagnosis. Those two topics were absent from the 550 SFT pairs. The fix is better training data, not more DPO. The Phase 11 rubric is calibrated to catch exactly these failure modes. The Phase 12 RAG layer is designed to fix the underlying knowledge gap. Every phase feeds the next.

**Failure modes documented:**
- Transformers 5.x incompatible with `eos_token_id` list type — fix: index with `[0]` defensively
- DPOTrainer in trl 0.9.6 requires `DPOConfig`, not `TrainingArguments`
- Adapter swapping on live PEFT model creates nested `model.model.model` hierarchy with silently wrong weights — always reload base model fresh for each adapter
- HuggingFace Hub uploads include 336MB optimizer checkpoints by default — use `ignore_patterns` to exclude

---

## Phase 11 — Evals: Measuring Model Behavior Before You Build on It

**Hardware:** A100 SXM4 80GB, CUDA 12.4, PyTorch 2.4.0+cu121, transformers 4.44.0, RunPod
**Models evaluated:** Llama 3.1 8B BASE, Phase 9 SFT adapter (r=16), Phase 10 DPO adapter
**Judge model:** claude-sonnet-4-20250514
**Note:** 39% inference throughput penalty observed between BASE (18.1 tok/s) and adapter variants (11.0-11.1 tok/s) from PEFT overhead. Production fix: `merge_and_unload()` before serving.

### Key Results

**Contamination Test**

10 original vs. rephrased prompt pairs. Average score delta: +0.15. Contamination signals: 0/10.

Interpretation: the model is not pattern-matching surface form. Hallucination is confabulation from partial knowledge generated in real time, not memorized wrong answers. This is harder to fix than contamination — rephrase the question a dozen ways and you get the same confidently wrong answer every time. The correct remediation is RAG grounding and an Editor gate, not data decontamination.

**Rubric Design and Calibration**

Four dimensions targeting the Phase 10 failure mode (confident hallucination amplified by SFT):

| Dimension | What It Catches |
|-----------|----------------|
| Technical Accuracy | Wrong definitions, wrong numbers, wrong mechanisms |
| Calibration | Confident hallucination — correct confidence requires correct knowledge |
| Mechanistic Depth | Vague explanations that sound correct but explain nothing |
| Audience Calibration | Wrong depth for a practitioner audience |

Deliberately excluded: length, fluency, formatting. These are what naive judges reward and what SFT optimized for. They are not correlated with technical correctness on niche AI infrastructure topics.

Calibration: 20 responses manually scored and compared against judge scores.

| Dimension | Pearson r | MAE | Status |
|-----------|-----------|-----|--------|
| technical_accuracy | 0.808 | 0.40 | PASS |
| calibration | 0.829 | 0.40 | PASS |
| mechanistic_depth | 0.720 | 0.55 | NEEDS REVIEW |
| audience_calibration | 0.811 | 0.90 | PASS |
| OVERALL | 0.861 | 0.562 | PASS |

Judge scored higher than human on 17/20 responses (85% positive bias). Root cause: score-2 anchor on technical accuracy was too forgiving. Rubric v1.1 tightened: a response that gets the core definition wrong scores 2 or below on technical_accuracy regardless of how well the rest reads. Overall r=0.861 passes the 0.75 threshold — judge is trustworthy for autonomous Editor decisions.

**Model Comparison: 50 Prompts, 3 Variants, 150 Total Responses**

| Variant | Mean Score | Confident Hallucination Flags |
|---------|-----------|-------------------------------|
| BASE | 2.400 | 27/50 (54%) |
| SFT | 2.520 | 37/50 (74%) |
| DPO | 2.575 | 37/50 (74%) |

Per-dimension breakdown:

| Dimension | BASE | SFT | DPO | Direction |
|-----------|------|-----|-----|-----------|
| technical_accuracy | 2.020 | 1.820 | 1.920 | Fine-tuning made accuracy worse |
| calibration | 2.460 | 2.040 | 2.060 | Fine-tuning made calibration worse |
| mechanistic_depth | 2.040 | 2.720 | 2.760 | Fine-tuning helped significantly |
| audience_calibration | 3.080 | 3.500 | 3.560 | Fine-tuning helped significantly |

Clean two-by-two: fine-tuning improved the style dimensions and degraded the accuracy dimensions. The model learned to write like an expert without becoming one. SFT taught confident, structured, practitioner-appropriate phrasing — it did not teach the underlying technical facts, because those facts were sparse in 550 training pairs.

Notable findings: SFT scored 1.964 on agent_infrastructure vs. BASE at 2.679 — the largest single-category regression. The SFT dataset had minimal agent infrastructure coverage; the model applied confident phrasing patterns from topics it knew to questions it knew least about. No variant averaged above 2.6/5. No variant is reliable enough to publish without an Editor gate.

**Editor Agent Decision Function (Phase 16)**

```
APPROVE if:
  technical_accuracy >= 3.0
  AND calibration >= 3.0
  AND mean_score >= 3.25
  AND judge_flag != "confident_hallucination"
REJECT otherwise — return dimension scores and reasoning to Writer
Maximum 3 revision cycles before escalating to human review
```

Technical accuracy and calibration are weighted double in the approval gate — these are the dimensions where DPO is actively unreliable. Any confident_hallucination flag is automatic reject regardless of mean score.

### What This Means

The contradiction between Phase 10 (BASE scored highest at 2.30) and Phase 11 (BASE ranks lowest in mean at 2.40 but worst on style dimensions) resolves cleanly: Phase 10 used a generic helpfulness rubric that rewarded structured confident responses. Phase 11 explicitly penalizes confident hallucination. Same model, different rubric, opposite ranking. The rubric determines what you measure. Helpful-sounding and technically correct are not the same thing, and no benchmark distinguishes them automatically.

The contamination finding changes the remediation strategy. If hallucination were contamination, the fix would be data decontamination. Since it is confabulation, the fix is giving the model access to correct information at inference time — which is exactly what Phase 12 builds.

### Key Insight

Define the quality bar before building the system that must maintain it. The rubric calibrated here is not an afterthought — it is a load-bearing component of the capstone. An uncalibrated judge inside an autonomous feedback loop reinforces the failure modes it was supposed to catch. The calibration methodology (score 20 manually, compute Pearson r, identify divergence pattern, tighten rubric anchors) is the correct engineering response and takes under an hour. Most teams skip it entirely. The difference is the difference between an Editor agent that works and one that approves its own hallucinations.

---
---

## Phase 12 — RAG + Storage Architecture

**Hardware:** M1 Pro, 16GB unified memory (all benchmarks local, all three backends running simultaneously)
**Models:** all-MiniLM-L6-v2 (22M params), all-mpnet-base-v2 (110M params), ms-marco-MiniLM-L-6-v2 cross-encoder
**Storage backends:** Chroma (in-process), PostgreSQL 16 + pgvector 0.8.2 (IVFFlat index), CockroachDB 26.1.2 + pgvector (sequential scan — vector indexing not yet supported in this version)
**Eval framework:** RAGAS 0.4.3, GPT-4o-mini as judge
**Corpus:** Phase 1-11 notes, 11 files, 106 chunks (full-stack-view synthesis document excluded — see Exercise 12.2)

### Key Results

**Exercise 12.1 — Embeddings From First Principles**

Tested MiniLM (22M) and MPNet (110M) against three pair categories drawn from the Phase 1-11 corpus: within-group similarity, adversarial pairs (same term, different semantic context), and paraphrase pairs (different vocabulary, same meaning).

| Metric | MiniLM | MPNet | Winner |
|--------|--------|-------|--------|
| Within-group similarity (Flash Attention) | 0.430 | 0.420 | Tie |
| Paraphrase: "HBM round trips" vs "memory access overhead" | 0.510 | 0.653 | MPNet |
| Adversarial: paper title vs Flash Attention description | 0.124 | 0.374 | MiniLM |
| Cross-group separation ratio (FA vs Distributed Training) | 2.0x | 1.9x | Tie |

Model selected: all-mpnet-base-v2. The paraphrase improvement (+0.143) outweighs the adversarial risk. Vocabulary variation between note prose and agent queries is the higher-frequency production failure mode. The adversarial result is a known risk to design around, not a reason to choose a weaker model.

The 2x separation ratio between within-group and cross-group similarity means retrieval works but admits noise. At 5x+ the retrieval is clean. This gap established the chunking requirement for Exercise 12.2: each chunk must encode one coherent concept.

**Exercise 12.2 — RAG From Scratch**

Complete pipeline without any framework: fixed-size chunking (400 words, 50-word overlap), MPNet embedding, SimpleVectorStore (numpy dot product), top-5 retrieval, Claude Haiku generation. 10 queries across four categories.

Note: full-stack-view synthesis document excluded from corpus. Initially included, it inflated retrieval scores by providing vocabulary-rich summaries that competed with primary sources and masked chunking failures. Test against primary sources only.

| Query Type | Hit Rate | Top-1 Accuracy | Queries |
|------------|----------|----------------|---------|
| Direct | 100% | 100% | 5 |
| Paraphrase | 75% | 50% | 2 |
| Cross-document | 50% | 0% | 1 |
| Abstract | 100% | 100% | 2 |
| **Overall** | **85%** | **80%** | **10** |

**Query 7 — the critical failure mode.** "Why did DPO replace PPO-based RLHF in practice?" — hit rate 100% (right source retrieved), generation answer: "The context does not contain this information." The explanation exists in the notes but was split across a chunk boundary by fixed-size chunking. Neither half was complete enough to answer. Hit rate showed success. The model correctly refused to hallucinate. The failure was invisible until RAGAS.

**Query 4 — flat score distribution.** Top-5 scores ranged 0.504 to 0.477 — a 0.027 spread. Query vocabulary matched hardware architecture language as strongly as inference language. No clean separation. Target for HyDE in Exercise 12.4.

**Exercise 12.3 — Storage Benchmark**

Four metrics per backend: single-writer ingestion throughput, 4-concurrent-writer throughput, similarity query latency p50/p99, hybrid query latency p50/p99 (vector similarity + SQL predicate in one transaction).

| Backend | Single Writer | 4 Concurrent Writers | Sim p50 | Sim p99 | Hybrid p50 | Hybrid p99 |
|---------|-------------|---------------------|---------|---------|-----------|-----------|
| Chroma | 1,258 doc/s | 1,155 doc/s | 0.5ms | 0.7ms | 0.6ms | 0.8ms |
| pgvector/Postgres | 717 doc/s | 666 doc/s | 3.7ms | 7.7ms | 7.5ms | 12.0ms |
| pgvector/CockroachDB | 383 doc/s | 640 doc/s | 3.7ms | 4.3ms | 3.5ms | 4.4ms |

Key findings:

**Chroma is fastest everywhere and the wrong choice for the capstone.** The 0.5ms vs 3.7ms gap is the cost of correctness. Chroma's hybrid "filter" runs post-retrieval in Python — at 106 chunks it looks like a real hybrid query. At 10 million chunks it retrieves thousands of candidates to filter down to 5. pgvector executes the SQL predicate inside the index scan.

**CockroachDB p99 is tighter than Postgres despite no vector index.** Postgres similarity p99: 7.7ms. CockroachDB: 4.3ms. The IVFFlat index at 106 vectors hurts more than it helps — the corpus is too small. CockroachDB's consistent execution produces lower tail latency at this scale.

**CockroachDB concurrent write speedup: 1.67x.** Single writer: 383 doc/s. Four concurrent writers: 640 doc/s. Postgres degraded under concurrency (0.93x). CockroachDB's distributed architecture parallelizes transaction processing across internal range partitioning — it gets faster under concurrent load rather than slower. This is the property that matters when four agents write to shared state simultaneously.

Storage architecture decision tree:
- Prototype, single agent, no complex predicates: Chroma
- Single-node production with relational query requirements: pgvector/Postgres
- Multi-agent concurrent writes, consistency under failure, horizontal scale: pgvector/CockroachDB

The capstone is the third case.

**Exercise 12.4 — Advanced RAG: HyDE + Reranking**

Two retrieval improvements measured independently and combined against the Exercise 12.2 baseline.

**HyDE:** Generate a hypothetical answer with Claude Haiku, embed that, search with the generated embedding. The hypothesis uses document vocabulary and lands in the right semantic neighborhood rather than sitting ambiguously between them.

**Reranking:** Retrieve top-20 candidates with embedding similarity, score each (query, chunk) pair jointly with ms-marco-MiniLM cross-encoder, return top-5 from reranked results.

| Method | Hit Rate | Top-1 Accuracy |
|--------|----------|----------------|
| Baseline | 85% | 80% |
| HyDE | 100% | 90% |
| Rerank | 85% | 60% |
| HyDE + Rerank | 85% | 60% |

HyDE fixed Query 4 exactly as predicted. The generated hypothesis — "Memory bandwidth constraints, rather than computational capacity, represent the primary limitation..." — used inference infrastructure vocabulary that landed cleanly in the right neighborhood.

The reranker hurt top-1 accuracy from 80% to 60%. The ms-marco cross-encoder was trained on web search query-document pairs and penalized dense technical jargon. A domain-fine-tuned reranker would reverse this result. General rerankers can actively hurt retrieval quality on technical corpora.

**Capstone decision: HyDE only, no reranker.** HyDE: +15% hit rate, +10% top-1. Reranker: flat hit rate, -20% top-1. The data makes the choice.

**Exercise 12.5 — RAG Evaluation with RAGAS**

Formal RAGAS evaluation on the same 10-query test set. Run twice to measure stability.

| Metric | Run 1 | Run 2 | Notes |
|--------|-------|-------|-------|
| Faithfulness | 0.979 | 0.958 | Near-perfect, stable |
| Answer relevancy | 0.552 | 0.722 | Variance from RAGAS internal sampling |
| Context precision | 0.911 | 0.911 | Identical across runs |
| Context recall | 0.950 | 0.950 | Identical across runs |

Query 7 RAGAS breakdown:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Hit rate | 100% | Right source retrieved |
| Faithfulness | 1.000 | Model correctly refused to hallucinate |
| Answer relevancy | 0.000 | Answer was useless — didn't address the question |
| Context recall | 0.500 | Key information missing from retrieved chunks |

Faithfulness 1.000 and answer relevancy 0.000 on the same query is the signature of a chunking failure: correct retrieval, incomplete chunk, correct refusal to hallucinate, useless answer. Hit rate cannot distinguish this from a success. RAGAS can.

Answer relevancy variance (+0.170 between runs) reflects LLM sampling in the RAGAS evaluation prompt. Context precision and recall are stable. For production eval pipelines, run 3+ times and average the volatile metrics.

### What This Means

Every RAG framework is the Exercise 12.2 pipeline with abstractions. Building it from scratch means every framework abstraction is legible and every framework failure is diagnosable. The failure mode hierarchy is what matters for the capstone: chunking failures produce silent non-answers — the model retrieves correctly, refuses to hallucinate, and returns nothing useful. Without the Editor agent's eval gate these publish. This is why the eval harness was built in Phase 11 before the agents.

The storage decision is the one the benchmark was designed to make visible. Chroma is fastest by every metric and the wrong choice when four agents write to shared state under consistency requirements. The 1.67x CockroachDB concurrent write improvement is the distributed SQL property in action — it gets faster under concurrent load rather than slower. The 3x ingestion gap vs. Chroma is the cost of that correctness.

### Key Insight

The most expensive mistake in a RAG pipeline is optimizing retrieval metrics while ignoring generation metrics. Query 7 had 100% hit rate and 0% answer relevancy. Every team that ships a RAG pipeline without RAGAS or an equivalent evaluation layer is measuring the wrong thing. Hit rate is necessary, not sufficient. The chunking decision — which most teams treat as a default parameter — is the actual retrieval architecture. Invest in chunking before tuning embeddings, index types, or retrieval strategies.
---

## Notes
 
- [Modern LLM Architecture: nanoGPT vs. Llama 3.1 8B](notes/modern-llm-architecture.md) — Phase 8 complete: architecture comparison table (all numbers from loaded model), KV cache arithmetic across MHA vs. GQA vs. MQA, MoE routing analysis v1 and v2
- [Fine-Tuning Mental Model: SFT, LoRA, QLoRA](notes/fine-tuning-mental-model.md) — Phase 9 complete: how SFT/LoRA/QLoRA stack as layers not alternatives, QLoRA end-to-end on Llama 3.1 8B, rank sensitivity experiment r=4 to r=64 (no plateau on technical domain data), qualitative hallucination correction and new failure mode from Socratic training format
- [Alignment Techniques: RLHF, DPO, Reward Modeling](notes/alignment-techniques.md) — Phase 10 complete: reward model training on HH-RLHF (0.75 accuracy, 1.46 margin), DPO on SFT adapter (0.58 accuracy, 0.44 margin), three-way qualitative comparison (BASE/SFT/DPO), LLM-as-judge scoring with counterintuitive BASE > SFT+DPO finding, failure modes: adapter stacking, DPOConfig API, eos_token_id list type
- [Evals: Measuring Model Behavior](notes/evals-mental-model.md) — Phase 11 complete: contamination test (0/10 signals — hallucination is confabulation not memorization), rubric v1.1 calibrated to r=0.861 overall, 150 responses scored across 3 variants, clean two-by-two (fine-tuning improved style, degraded accuracy), Editor agent decision function defined and committed
- [Storage Architecture for Agentic AI](notes/storage-architecture-decision.md) — Phase 12 complete: MPNet selected on empirical paraphrase data, RAG from scratch (85% hit rate baseline), HyDE +15%/+10% improvement, reranker rejected on domain mismatch, RAGAS Query 7 proof case (100% hit rate / 0% answer relevancy), storage benchmark across Chroma/pgvector/CockroachDB with concurrent write results
- [Tool Use and the ReAct Pattern](notes/react-pattern.md) — Phase 13: raw function calling with the Anthropic API, the ReAct thought/action/observation loop, explicit state management across multi-turn agent workflows, failure modes when tools fail or the model loops
- [Agent Framework Comparison: LangGraph, OpenClaw, CrewAI](notes/agent-framework-comparison.md) — Phase 14: agents as state machines (LangGraph), OpenClaw architecture dissection (persistent memory, skills system, context management), CrewAI multi-agent orchestration, when to choose each framework
- [Production Agentic Infrastructure](notes/production-agentic-infra.md) — Phase 15: full agent tracing with LangSmith/Langfuse, cost modeling per agent run, retry/fallback/circuit breaker patterns, concurrent agent state under contention, prompt injection defense
- [Content Engine Architecture](notes/content-engine-architecture.md) — Phase 16 capstone: every component, every binding constraint, every agent handoff — the synthesis document for Part II
 
---
 
## Hardware
 
| Phase | Hardware | Provider | Cost |
|-------|----------|----------|------|
| 8 | A100 SXM4 80GB (Mixtral routing) | RunPod |
| 9 | A100 SXM4 80GB | RunPod | ~$0.15 (5m10s training run) |
| 10 | A100 SXM4 80GB | RunPod | ~$2.50 (reward model ~$2, DPO ~$0.50) |
| 11 | A100 SXM4 80GB | RunPod | ~$1.50 (inference + 150 judge API calls) |
| 12 | M1 Pro 16GB unified memory (all backends local) | Local | ~$0.50 (HyDE + RAGAS API calls) |
 
---
 
## Companion Repo
 
Part I — [github.com/dagc-ai/ai-infra-learning](https://github.com/dagc-ai/ai-infra-learning)
 
Seven phases. Silicon to transformer. CUDA kernels, Ring AllReduce from scratch, vLLM on A100s, quantization benchmarks, Groq vs. A100 head-to-head, and a 30M parameter GPT trained from scratch with deliberate failure modes engineered and documented. The foundation this curriculum builds on.
