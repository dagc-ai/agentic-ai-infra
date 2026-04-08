# RAG + Storage Architecture
## Phase 12 — Infrastructure for Agentic AI

---

## Goal

Build a complete retrieval-augmented generation pipeline from first
principles and make informed storage architecture decisions for
multi-agent systems. Understand embeddings at the mathematical level,
build every component of the RAG pipeline without frameworks so every
abstraction is legible, benchmark three storage backends under realistic
agent workloads, and validate the pipeline with formal evaluation metrics.

The guiding question: given a multi-agent system where four specialized
agents share state and retrieve knowledge simultaneously, what storage
architecture do you choose and why?

---

## Exercise 12.1 — Embeddings From First Principles

### What Was Done

Tested two embedding models (MiniLM-L6-v2 at 22M params and
all-mpnet-base-v2 at 110M params) against three categories of sentence
pairs drawn directly from the Phase 1-11 corpus:

- Within-group similarity (Flash Attention sentences, Distributed
  Training sentences, Alignment sentences)
- Adversarial pairs (same technical term, different semantic context)
- Paraphrase pairs (different vocabulary, same meaning)

### Results

| Metric | MiniLM | MPNet | Winner |
|--------|--------|-------|--------|
| Within-group similarity (FA) | 0.430 | 0.420 | Tie |
| Paraphrase: HBM round trips vs memory access overhead | 0.510 | 0.653 | MPNet |
| Adversarial: paper title vs Flash Attention | 0.124 | 0.374 | MiniLM |
| Cross-group separation ratio (FA vs DT) | 2.0x | 1.9x | Tie |

**Model selected: all-mpnet-base-v2.** The paraphrase improvement
(+0.143) outweighs the adversarial risk for this corpus. Vocabulary
variation between note prose and agent queries is the higher-frequency
failure mode in production.

### Implications

An embedding is a function that maps text to a point in high-dimensional
space such that semantically similar texts land near each other. The
geometry is learned -- only the direction of the vector encodes meaning,
not the magnitude. Cosine similarity measures the angle between vectors,
which is why normalized embeddings make dot product equivalent to cosine
similarity.

The separation ratio of 2x between within-group and cross-group
similarity means retrieval will surface relevant chunks but will admit
noise. A chunk that mixes multiple concepts embeds to an averaged
location that retrieves poorly for any specific query. This established
the chunking requirement for Exercise 12.2: each chunk must encode one
coherent concept.

The adversarial test revealed a meaningful tradeoff. MPNet scored
0.374 similarity between the paper title "Attention Is All You Need"
and a Flash Attention description -- almost pulling them into the same
neighborhood. MiniLM correctly separated them at 0.124. Bigger models
bridge vocabulary gaps better but are also more susceptible to surface-
level term overlap. For a technical corpus, knowing this failure mode
exists is the prerequisite for designing around it.

---

## Exercise 12.2 — RAG From Scratch

### What Was Done

Built a complete RAG pipeline without any framework. Every step
explicit: fixed-size chunking (400 words, 50-word overlap), MPNet
embedding, SimpleVectorStore (numpy dot product), top-5 retrieval,
Claude Haiku generation. Tested against 10 queries across four
categories on the Phase 1-11 corpus (11 files, 106 chunks after
removing the full-stack-view synthesis document).

The full-stack-view document was initially included and inflated
retrieval scores by providing vocabulary-rich summaries that competed
with primary sources. Removing it produced honest baseline numbers
and revealed that the summary document was masking chunking failures
rather than surfacing them.

### Results -- Clean Baseline (without full-stack-view)

| Query Type | Hit Rate | Top-1 Accuracy | Queries |
|------------|----------|----------------|---------|
| Direct | 100% | 100% | 5 |
| Paraphrase | 75% | 50% | 2 |
| Cross-document | 50% | 0% | 1 |
| Abstract | 100% | 100% | 2 |
| **Overall** | **85%** | **80%** | **10** |

### Key Failure Modes Identified

**Query 7 -- retrieval success, generation failure (the critical case).**
"Why did DPO replace PPO-based RLHF in practice?" retrieved the correct
source (hit rate 100%, top-1 correct) but returned the answer: "The
context does not contain this information." The explanation exists in
the notes but was split across a chunk boundary by fixed-size chunking.
Neither half was complete enough to answer the question. Hit rate
showed success. The model correctly refused to hallucinate.
This failure mode is invisible to hit rate metrics. It required RAGAS
to surface.

**Query 4 -- flat score distribution.**
"What keeps GPUs from being fully utilized during inference?" returned
five chunks with scores between 0.504 and 0.477 -- a 0.027 spread
across the top-5. The query vocabulary matched hardware architecture
language as strongly as inference language. No clean separation.
This is the target for HyDE in Exercise 12.4.

**Cross-document structural limitation.**
A single query vector points in one direction in embedding space. A
question requiring synthesis across two documents needs both
neighborhoods simultaneously. This is a recall problem -- not fixable
with better vocabulary matching, only with multi-vector retrieval or
larger top-K.

### Implications

Every RAG framework is this pipeline with abstractions. The ingestion
step, the embedding step, the index, the retrieval, the generation --
all present in LlamaIndex, LangChain, and Haystack. Building it from
scratch means every framework decision is legible when you use one.

The failure mode hierarchy matters for the capstone Researcher agent.
Chunking failures produce silent non-answers -- the model retrieves
correctly, refuses to hallucinate, and returns nothing useful. Without
an eval gate (the Editor agent using the Phase 11 rubric) these silent
failures publish. This is why the eval harness was built before the
agents.

---

## Exercise 12.3 — Storage Benchmark

### What Was Done

Benchmarked three storage backends under realistic agent workloads on
M1 Pro, 16GB unified memory:

- **Chroma** -- dedicated in-process vector store, no persistence
  guarantees
- **pgvector/Postgres** -- PostgreSQL 16 with pgvector 0.8.2, IVFFlat
  index, ACID guarantees
- **pgvector/CockroachDB** -- CockroachDB 26.1.2 with pgvector,
  no vector index (vector indexing not yet supported in this version),
  distributed SQL with Raft consensus

Metrics: single writer ingestion throughput, 4 concurrent writer
throughput, similarity query latency p50/p95/p99, hybrid query latency
p50/p95/p99 (vector similarity + SQL predicate in one transaction).

### Results

| Backend | SW (doc/s) | CW (doc/s) | Sim p50 | Sim p99 | Hyb p50 | Hyb p99 |
|---------|-----------|-----------|---------|---------|---------|---------|
| Chroma | 1,258 | 1,155 | 0.5ms | 0.7ms | 0.6ms | 0.8ms |
| pgvector/PG | 717 | 666 | 3.7ms | 7.7ms | 7.5ms | 12.0ms |
| pgvector/CRDB | 383 | 640 | 3.7ms | 4.3ms | 3.5ms | 4.4ms |

Hardware: M1 Pro, 16GB unified memory. All three backends local.
CockroachDB numbers reflect full sequential scan (no vector index).

### Key Findings

**Chroma is fastest everywhere and it is the wrong choice for the
capstone.** Chroma runs in-process with no network overhead, no
transaction log, no ACID guarantees. The 0.5ms vs 3.7ms gap is the
cost of correctness. Chroma's hybrid "filter" runs post-retrieval in
Python -- it retrieves candidates then filters in application code.
At 106 chunks this is indistinguishable from a real hybrid query.
At 10 million chunks you retrieve thousands of candidates to get 5
that pass the filter. pgvector executes the SQL predicate inside the
index scan -- only relevant chunks are ever touched.

**CockroachDB p99 is tighter than Postgres despite no vector index.**
Postgres similarity p99: 7.7ms. CockroachDB similarity p99: 4.3ms.
The IVFFlat index at 106 vectors hurts more than it helps -- the
corpus is too small for the index to be beneficial. CockroachDB's
consistent query execution produces lower tail latency on this corpus
size.

**CockroachDB concurrent write speedup: 1.67x.**
Single writer: 383 docs/sec. Four concurrent writers: 640 docs/sec.
Postgres went slightly slower under concurrency (0.93x). CockroachDB's
distributed architecture benefits from concurrent writes because it
parallelizes transaction processing across its internal range
partitioning. This is the property that matters when four agents
write to shared state simultaneously.

**CockroachDB vector indexing note.**
CockroachDB 26.1.2 does not support vector indexing (planned for a
future release). All vector queries run as full sequential scans.
At production scale with millions of vectors this would be the
binding constraint. For the capstone corpus at thousands of vectors,
sequential scan latency is acceptable.

### Storage Architecture Decision

The decision tree for agent systems:

- **Prototype, no multi-agent coordination, no complex predicates:**
  Chroma. Zero setup, fast, good enough.
- **Single-node production with relational query requirements:**
  pgvector/Postgres. ACID guarantees, hybrid queries, operational
  simplicity.
- **Multi-agent under concurrent write load, requiring consistency,
  availability under node failure, and horizontal scale:**
  pgvector/CockroachDB.

The capstone is the third case. Four agents write to shared state.
The article status, research notes, eval scores, and vector embeddings
all live in one database. A node failure during a write does not
corrupt agent state or leave workflows in an unknown intermediate
state. Raft consensus ensures every acknowledged write is durable.

The vector store is additive. The serializable isolation on agent
state tables is the core value. The fact that the same database
handles both eliminates the need for a separate vector store and a
separate relational store coordinated by application-level logic.

---

## Exercise 12.4 — Advanced RAG: HyDE + Reranking

### What Was Done

Implemented two retrieval improvements on top of the Exercise 12.2
baseline and measured each independently and combined:

**HyDE (Hypothetical Document Embeddings):** Instead of embedding
the raw query, generate a hypothetical answer to the query with
Claude Haiku, embed that, and search with the generated embedding.
The hypothesis uses document vocabulary and lands in the right
semantic neighborhood rather than sitting ambiguously between
neighborhoods.

**Cross-encoder reranking:** Retrieve top-20 candidates with
embedding similarity, then score each (query, chunk) pair jointly
with a cross-encoder model (ms-marco-MiniLM-L-6-v2). Return top-5
from reranked results. More precise than embedding similarity alone
but slower.

### Results vs Baseline

| Method | Hit Rate | Top-1 Accuracy |
|--------|----------|----------------|
| Baseline | 85% | 80% |
| HyDE | 100% | 90% |
| Rerank | 85% | 60% |
| HyDE + Rerank | 85% | 60% |

### Key Findings

**HyDE fixed Query 4 exactly as predicted.**
The flat-score query ("What keeps GPUs from being fully utilized
during inference?") went from 50% hit rate and wrong top-1 to 100%
hit rate and correct top-1. The hypothesis generated: "Memory
bandwidth constraints, rather than computational capacity, represent
the primary limitation..." -- inference infrastructure vocabulary that
landed squarely in the right neighborhood.

**The reranker hurt top-1 accuracy from 80% to 60%.**
The cross-encoder is a general-purpose model trained on web search
pairs. It penalized dense technical jargon and rewarded conversational
natural language -- the opposite of what a technical AI infrastructure
corpus needs. A domain-fine-tuned reranker would reverse this result.
General rerankers can actively hurt retrieval quality on domain-
specific technical corpora.

**Capstone decision: HyDE only, no reranker.**
The decision is empirical. HyDE: +15% hit rate, +10% top-1. Reranker:
0% hit rate change, -20% top-1. Data makes the choice.

---

## Exercise 12.5 — RAG Evaluation with RAGAS

### What Was Done

Ran formal RAGAS evaluation (Python 3.11, RAGAS 0.4.3, GPT-4o-mini
as judge) on the same 10-query test set from Exercise 12.2. Measured
four metrics independently: faithfulness, answer relevancy, context
precision, context recall. Ran twice to measure evaluation stability.

### Results (Two Runs)

| Metric | Run 1 | Run 2 | Notes |
|--------|-------|-------|-------|
| Faithfulness | 0.979 | 0.958 | Near-perfect, stable |
| Answer relevancy | 0.552 | 0.722 | Variance from RAGAS sampling |
| Context precision | 0.911 | 0.911 | Identical, stable |
| Context recall | 0.950 | 0.950 | Identical, stable |

### Query 7 -- The Proof Case

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Hit rate (Ex 12.2) | 100% | Right source retrieved |
| Faithfulness | 1.000 | Model correctly refused to hallucinate |
| Answer relevancy | 0.000 | Answer was useless -- didn't address question |
| Context recall | 0.500 | Key information missing from retrieved chunks |

Hit rate said the pipeline succeeded. RAGAS revealed it failed.
Faithfulness 1.000 and answer relevancy 0.000 in the same query
is the signature of a chunking failure: correct retrieval, incomplete
chunk, correct refusal to hallucinate, useless answer.

### What RAGAS Shows That Hit Rate Cannot

Hit rate is binary per query. It measures whether the right source
appeared in the top-5. It cannot distinguish between a retrieved
chunk that enabled a correct answer and a retrieved chunk that was
present but insufficient. RAGAS answer relevancy catches this gap.

The run-to-run variance on answer relevancy (+0.170 between runs)
reflects RAGAS internal sampling variance in how GPT-4o-mini
constructs evaluation prompts at temperature > 0. Context precision
and context recall were identical across both runs -- those are the
stable metrics. For production eval pipelines, run 3+ times and
average.

### Capstone Implication

The Editor agent's quality gate from Phase 11 is the production
equivalent of RAGAS answer relevancy. A RAG pipeline with perfect
faithfulness and poor answer relevancy produces confident non-answers
that look correct. Without the Editor gate those non-answers publish.
The eval harness is not optional infrastructure -- it is what
separates an autonomous system that works from one that publishes
garbage confidently.

---

## Phase 12 Synthesis

### What the Phase Built

A complete, evaluated RAG pipeline ready for integration into the
capstone content engine:

- Embedding model selected on empirical quality data (MPNet, paraphrase
  bridging 0.653)
- Chunking strategy established (400 words, one concept per chunk,
  split on conceptual boundaries)
- Retrieval baseline measured (85% hit rate, 80% top-1)
- HyDE improvement validated (+15% hit rate, +10% top-1)
- Reranker rejected on empirical grounds (domain mismatch)
- RAGAS eval integrated, Query 7 failure mode documented and
  understood
- Storage backend selected on benchmark data (pgvector/CockroachDB
  for capstone)

### The Binding Constraints at Each Layer

| Layer | Binding Constraint | Finding |
|-------|-------------------|---------|
| Embedding model | Paraphrase bridging vs adversarial precision | MPNet wins on paraphrase (+0.143), acceptable adversarial risk |
| Chunking | Conceptual completeness vs retrieval precision | Fixed-size splits create silent generation failures |
| Retrieval | Vocabulary match between query and document | HyDE resolves by searching document-space with a document |
| Reranking | Domain alignment of the reranker | General rerankers actively hurt technical corpora |
| Storage | Consistency vs performance vs operational simplicity | CockroachDB for multi-agent, Postgres for single-agent production |
| Evaluation | What hit rate misses | Faithfulness + answer relevancy together catch chunking failures |

---

## Technical Insights

**Embeddings encode direction, not magnitude.** Two chunks about Flash
Attention point in roughly the same direction from the origin regardless
of their length. Cosine similarity measures that angle. Normalizing
vectors first makes dot product equivalent to cosine similarity --
which is why embedding models return normalized vectors and why the
`<=>` operator in pgvector gives you what you want.

**The separation ratio is the retrieval quality signal.** At 2x
separation between within-group and cross-group similarity, retrieval
works but admits noise. At 5x+, retrieval is clean. The gap between
where you are and where you want to be tells you whether to invest in
better embeddings, better chunking, or retrieval augmentation like HyDE.

**IVFFlat indexes require sufficient data to be beneficial.** At 106
vectors the index introduced more overhead than it saved. CockroachDB's
sequential scan outperformed Postgres's indexed scan at p99 on this
corpus. Index benefits emerge at thousands of vectors. Below that,
exact search with a sequential scan is both faster and more accurate.

**HyDE works because queries and documents live in different vocabulary
spaces.** A user asks "What keeps GPUs underutilized?" Your documents
say "inference is memory-bandwidth bound at small batch sizes." HyDE
generates the latter from the former, then searches with it. You are
searching document-space with a document rather than query-space with
a question.

**General rerankers are trained on web search, not technical prose.**
The ms-marco cross-encoder was trained on Microsoft BING query-document
pairs. It learned that confident, specific, jargon-heavy responses score
lower than accessible natural language -- the opposite of what a
technical AI infrastructure corpus needs. Domain mismatch in the
reranker is worse than no reranker at all.

**CockroachDB concurrent write scaling (1.67x) is the distributed SQL
property in action.** Single-node Postgres serializes concurrent writes
through its lock manager -- four writers contend for the same resources
and throughput drops. CockroachDB distributes transactions across
internal ranges and processes them in parallel. The property that makes
it worth the operational complexity is exactly this: it gets faster
under concurrent load rather than slower.

---

## Developer Insights

**Build RAG from scratch before touching any framework.** The 50-line
SimpleVectorStore class is what Chroma, LlamaIndex, and LangChain are
doing underneath. Once you've built it, every framework abstraction
is legible and every framework failure is diagnosable. The investment
is one exercise. The return is permanent.

**Chunking is the retrieval architecture.** More than embedding model
choice, more than index type, more than retrieval strategy -- how you
split documents determines what is retrievable. A chunk that breaks a
sentence mid-explanation will never retrieve well regardless of what
you do downstream. Invest in chunking before tuning anything else.

**Measure retrieval and generation separately.** Hit rate measures
retrieval. RAGAS faithfulness and answer relevancy measure generation.
Query 7 scored 100% on hit rate and 0.000 on answer relevancy -- the
pipeline looked successful and returned a useless answer. You need
both measurement layers or you will ship silent failures.

**Run RAGAS multiple times and average.** Context precision and context
recall are stable across runs. Answer relevancy varies due to LLM
sampling in the evaluation prompt. A single run is not sufficient for
production decisions. Three runs and an average give you a reliable
signal.

**Remove synthetic summary documents from your RAG corpus.** The
full-stack-view document inflated retrieval scores by providing
vocabulary-rich summaries that competed with primary sources. It masked
chunking failures rather than surfacing them. Retrieval quality metrics
against a contaminated corpus are optimistic lies. Test against primary
sources only.

**The storage decision is irreversible at scale.** Migrating a
production vector store from Chroma to pgvector after you have millions
of embeddings and a live agent pipeline is a significant engineering
project. Make the decision based on your consistency requirements before
you have data, not after.

---

## Business Insights

**The vector store is not the differentiator -- the state management is.**
Every AI infrastructure vendor is selling vector search. The hard problem
is not storing and retrieving embeddings. The hard problem is maintaining
consistent agent state across concurrent writes, surviving infrastructure
failures without corrupting workflow state, and recovering cleanly when
something goes wrong. That problem requires distributed SQL, not a
dedicated vector database.

**Silent failures are the production risk that benchmarks don't show.**
A RAG pipeline with 100% hit rate and 0% answer relevancy on a specific
query class will publish non-answers autonomously. No alarm fires. No
error log entry. The Editor agent's eval gate is the only mechanism
that catches this. Teams that deploy RAG pipelines without calibrated
eval harnesses are running blind.

**Corpus quality determines RAG quality -- not model size.**
The contamination experiment showed that a summary document inflated
retrieval scores by masking failures. In enterprise deployments, the
corpus is often a mixture of authoritative documentation, marketing
copy, outdated internal wikis, and support tickets. The quality of
what you ingest determines the quality of what you retrieve. A strong
embedding model cannot compensate for a poorly curated corpus.

**The reranker finding has procurement implications.**
Organizations buying AI infrastructure often assume that more
sophisticated components produce better results. The cross-encoder
reranker cost 20 percentage points of top-1 accuracy on a technical
corpus. More expensive is not better when the component's training
distribution doesn't match your use case. Empirical benchmarking on
your specific data is the only reliable evaluation method.

**CockroachDB's value compounds with agent complexity.**
A single agent running on Postgres is fine. Two agents with shared
state on Postgres is manageable. Four agents with concurrent writes,
complex state transitions, and uptime requirements on single-node
Postgres is a production incident waiting to happen. The infrastructure
decision that looks over-engineered at prototype stage is the one that
prevents the 3am incident six months later.

---

## GTM Insights

**For AI hardware companies (Tenstorrent, Cerebras, Groq):**
The RAG pipeline is the primary inference workload for enterprise
AI applications today. Every enterprise RAG deployment runs embedding
generation, vector similarity search, and LLM generation on every
query. Understanding which of these steps is the binding constraint
at the hardware level -- embedding throughput, memory bandwidth for
attention during generation, or the latency of the retrieval call --
is the conversation that separates technical sellers from spec-sheet
readers. HyDE adds an extra generation call per query. At scale,
that additional latency is the hardware vendor's opportunity.

**For inference and serving platforms (Together AI, Fireworks, Modal):**
The RAG pipeline has two LLM calls when HyDE is active -- one for
hypothesis generation and one for answer generation. Enterprise
customers running high-volume RAG are paying for both. A seller who
can explain why HyDE improves retrieval quality, what the latency
cost is, and when the quality improvement justifies the additional
inference call is having a different conversation than one selling
tokens per second. The storage benchmark numbers -- 3.7ms retrieval
latency at p50 -- also matter for end-to-end latency budgets that
include retrieval as a first-class concern.

**For cloud database vendors (CockroachDB, PlanetScale, Neon):**
The benchmark proves the architecture argument empirically. Chroma
is 1258 docs/sec vs CockroachDB's 383 docs/sec for single-writer
ingestion. That gap is the cost of correctness. The winning GTM
conversation is not "our database is faster" -- it clearly isn't for
vector workloads. It is "your agent pipeline will eventually have a
3am incident caused by corrupted state under concurrent writes, and
here is the benchmark that shows you why, and here is the architecture
that prevents it." That is a consultative sale grounded in real
numbers, not a spec sheet comparison.

**For AI application platforms (LangChain, LlamaIndex, Cohere):**
The RAG-from-scratch exercise demonstrates exactly what these
frameworks abstract. A seller at a framework company who can walk
a customer through the SimpleVectorStore implementation -- showing
them what the framework does, why the abstractions exist, and where
they break down -- is demonstrating genuine technical depth rather
than product familiarity. The chunking failure mode (Query 7:
100% hit rate, 0% answer relevancy) is also a direct argument for
better framework tooling -- default chunking strategies in most
frameworks would produce this exact failure on this exact corpus.

**The one-liner for any enterprise AI conversation:**
"Most RAG implementations look correct in development and fail
silently in production. I know what the failure modes are because
I built the pipeline from scratch, measured it with formal evaluation
metrics, and documented where hit rate lied to me. Here is the number:
100% retrieval hit rate, 0% answer relevancy on the same query. That
is the failure your Editor agent needs to catch before anything
publishes."

---

*Phase 12 complete. All artifacts committed to*
*github.com/dagc-ai/agentic-ai-infra/phase12*
*Next: Phase 13 -- Tool Use and the ReAct Pattern*
