# Phase 12 — RAG + Storage Architecture

## Goal
Build a complete retrieval-augmented generation pipeline from scratch.
Make informed storage architecture decisions for multi-agent systems.

## Exercises
- 12.1 Embeddings from first principles — semantic geometry, model size vs quality
- 12.2 RAG from scratch — ingestion, chunking, retrieval, generation, no framework
- 12.3 Storage benchmark — Chroma vs pgvector/Postgres vs pgvector/CockroachDB
- 12.4 Advanced RAG — HyDE, cross-encoder reranking, measured improvement
- 12.5 RAG evaluation — RAGAS, retrieval metrics, generation metrics

## Corpus
Phase 1-11 notes from github.com/dagc-ai/agentic-ai-infra/notes/

## Hardware
Exercises 12.1, 12.2, 12.4, 12.5: M1 Pro (no GPU required)
Exercise 12.3 (storage benchmark): M5 Max (benchmark numbers must reflect capstone hardware)

## Results
| Exercise | Key Result | Commit |
|----------|-----------|--------|
| 12.1 | | |
| 12.2 | | |
| 12.3 | | |
| 12.4 | | |
| 12.5 | | |

# Embeddings Analysis — Phase 12.1

## Models Tested
- all-MiniLM-L6-v2 (22M params, 384 dims)
- all-mpnet-base-v2 (110M params, 768 dims)

## Key Results

| Metric | MiniLM | MPNet | Winner |
|--------|--------|-------|--------|
| FA group mean (within-group similarity) | 0.430 | 0.420 | Tie |
| Paraphrase: HBM round trips vs memory access overhead | 0.510 | 0.653 | MPNet |
| Adversarial: paper title vs Flash Attention | 0.124 | 0.374 | MiniLM |
| Cross-group separation ratio (FA vs DT) | 2.0x | 1.9x | Tie |

## Model Selection: all-mpnet-base-v2

MPNet selected for the RAG pipeline. The paraphrase improvement (+0.143)
outweighs the adversarial risk for this corpus. Vocabulary variation between
note prose and agent queries is the higher-frequency failure mode.

## Chunking Strategy Implications

**Separation ratio of ~2x is workable but requires clean chunks.**
A 2x separation between within-group and cross-group similarity means retrieval
will surface relevant chunks but will also admit noise. Chunk quality is the
primary lever for improving retrieval precision.

**Alignment group looseness (mean 0.199-0.244) reveals a chunking requirement.**
The alignment notes cover DPO mechanics, Bradley-Terry loss, reward modeling,
and Constitutional AI in the same documents. The embedding model correctly
spreads these apart -- they are related at the domain level but distinct
at the concept level. Chunks that mix multiple alignment concepts will embed
to an averaged location that retrieves poorly for any specific query.

**Chunking rules for Phase 12.2:**
- Target chunk size: 300-500 tokens
- Split on conceptual boundaries, not fixed character counts
- Each chunk should encode one coherent idea
- Overlap: 50 tokens to prevent context loss at boundaries

## Failure Mode: Vocabulary Mismatch
The paraphrase test (0.510-0.653) confirms the model can bridge vocabulary
gaps -- "HBM round trips" and "memory access overhead" are recognized as
equivalent. This is the primary retrieval risk for this corpus: notes use
precise ML vocabulary, queries may not.

HyDE (Phase 12.4) is the mitigation: generate a hypothetical answer in
document vocabulary, embed that, search with it instead of the raw query.