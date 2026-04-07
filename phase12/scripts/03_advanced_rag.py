"""
Exercise 12.4 -- Advanced RAG: HyDE + Reranking
Measures improvement over the Exercise 12.2 baseline.

Two techniques:
1. HyDE (Hypothetical Document Embeddings)
   - Generate a hypothetical answer to the query
   - Embed that answer instead of the raw query
   - Search document-space with a document, not query-space with a question

2. Cross-encoder reranking
   - Take top-20 from embedding retrieval
   - Score each (query, chunk) pair jointly with a cross-encoder
   - Return top-5 from reranked results
   - More precise than embedding similarity alone

Baseline for comparison: Exercise 12.2 clean run (85% hit rate, 80% top-1)
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
CORPUS_DIR    = Path("../data/corpus")
RESULTS_DIR   = Path("../data/results")
EMBED_MODEL   = "all-mpnet-base-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
TOP_K         = 5
RERANK_POOL   = 20   # retrieve this many, rerank down to TOP_K

# ── Reuse data structures and functions from 02_rag_from_scratch ──────────────
@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    word_count: int

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float

class SimpleVectorStore:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.chunks: List[Chunk] = []

    def add(self, chunk: Chunk, embedding: np.ndarray):
        self.embeddings.append(embedding)
        self.chunks.append(chunk)

    def search(self, query_embedding: np.ndarray, k: int = TOP_K) -> List[RetrievalResult]:
        if not self.embeddings:
            return []
        matrix = np.array(self.embeddings)
        scores = matrix @ query_embedding
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_k_indices
        ]

    def __len__(self):
        return len(self.chunks)

def load_corpus(corpus_dir: Path) -> Dict[str, str]:
    corpus = {}
    for path in sorted(corpus_dir.glob("*.md")):
        with open(path, "r", encoding="utf-8") as f:
            corpus[path.name] = f.read()
    return corpus

def chunk_fixed(text: str, source_file: str,
                chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    words = text.split()
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(Chunk(
            text=" ".join(words[start:end]),
            source_file=source_file,
            chunk_index=idx,
            word_count=end - start,
        ))
        if end == len(words):
            break
        start += chunk_size - overlap
        idx += 1
    return chunks

def build_index(corpus: Dict[str, str],
                model: SentenceTransformer) -> Tuple[SimpleVectorStore, List[Chunk]]:
    all_chunks = []
    for filename, content in corpus.items():
        all_chunks.extend(chunk_fixed(content, filename))

    embeddings = model.encode(
        [c.text for c in all_chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    store = SimpleVectorStore()
    for chunk, embedding in zip(all_chunks, embeddings):
        store.add(chunk, embedding)
    return store, all_chunks

def evaluate_retrieval(results: List[RetrievalResult],
                       expected_sources: List[str]) -> Dict:
    retrieved_sources = [r.chunk.source_file for r in results]
    hits = sum(1 for s in expected_sources if s in retrieved_sources)
    return {
        "retrieved_sources": retrieved_sources,
        "hits": hits,
        "hit_rate": hits / len(expected_sources) if expected_sources else 0,
        "top1_correct": retrieved_sources[0] in expected_sources if retrieved_sources else False,
    }

# ── Technique 1: HyDE ─────────────────────────────────────────────────────────
def hyde_retrieve(
    query: str,
    store: SimpleVectorStore,
    embed_model: SentenceTransformer,
    llm_client: anthropic.Anthropic,
    k: int = TOP_K,
) -> Tuple[List[RetrievalResult], float, str]:
    """
    HyDE retrieval:
    1. Ask Claude to generate a hypothetical answer to the query
    2. Embed the hypothetical answer (not the query)
    3. Search with that embedding

    Why this works: the hypothetical answer uses document vocabulary.
    "What keeps GPUs underutilized?" becomes a paragraph about
    memory-bandwidth-bound inference at small batch sizes -- language
    that lands in the right semantic neighborhood.
    """
    start = time.time()

    hyde_prompt = f"""Write a short technical paragraph (3-5 sentences) that directly 
answers the following question about AI infrastructure. Use precise technical 
vocabulary. Do not hedge or say you don't know -- write as if you are certain.

Question: {query}

Technical answer:"""

    response = llm_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": hyde_prompt}]
    )
    hypothetical_answer = response.content[0].text.strip()

    hyp_embedding = embed_model.encode(
        hypothetical_answer,
        normalize_embeddings=True
    )
    results = store.search(hyp_embedding, k=k)
    latency_ms = (time.time() - start) * 1000

    return results, latency_ms, hypothetical_answer

# ── Technique 2: Reranking ────────────────────────────────────────────────────
def rerank_retrieve(
    query: str,
    store: SimpleVectorStore,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    pool_size: int = RERANK_POOL,
    k: int = TOP_K,
) -> Tuple[List[RetrievalResult], float]:
    """
    Reranking pipeline:
    1. Retrieve top-20 candidates via embedding similarity (fast, coarse)
    2. Score each (query, chunk) pair with cross-encoder (slow, precise)
    3. Return top-5 by reranker score

    Why two stages? Cross-encoder reads query and document together --
    far more precise than embedding similarity. But it's O(n) inference
    calls, not one matrix multiply. Running it on all 106 chunks would
    be too slow. Running it on 20 candidates is fast enough.
    """
    start = time.time()

    query_embedding = embed_model.encode(query, normalize_embeddings=True)
    candidates = store.search(query_embedding, k=pool_size)

    pairs = [(query, c.chunk.text) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    results = [
        RetrievalResult(chunk=r.chunk, score=float(score))
        for r, score in reranked
    ]
    latency_ms = (time.time() - start) * 1000

    return results, latency_ms

# ── Technique 3: HyDE + Reranking combined ───────────────────────────────────
def hyde_rerank_retrieve(
    query: str,
    store: SimpleVectorStore,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    llm_client: anthropic.Anthropic,
    pool_size: int = RERANK_POOL,
    k: int = TOP_K,
) -> Tuple[List[RetrievalResult], float, str]:
    """
    Combined: HyDE for candidate retrieval, reranker for final ranking.
    HyDE gets you into the right neighborhood.
    Reranker picks the best chunks within that neighborhood.
    """
    start = time.time()

    hyde_prompt = f"""Write a short technical paragraph (3-5 sentences) that directly 
answers the following question about AI infrastructure. Use precise technical 
vocabulary. Do not hedge -- write as if you are certain.

Question: {query}

Technical answer:"""

    response = llm_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": hyde_prompt}]
    )
    hypothetical_answer = response.content[0].text.strip()

    hyp_embedding = embed_model.encode(
        hypothetical_answer,
        normalize_embeddings=True
    )
    candidates = store.search(hyp_embedding, k=pool_size)

    pairs = [(query, c.chunk.text) for c in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:k]

    results = [
        RetrievalResult(chunk=r.chunk, score=float(score))
        for r, score in reranked
    ]
    latency_ms = (time.time() - start) * 1000

    return results, latency_ms, hypothetical_answer

# ── Test queries (same as Exercise 12.2 for direct comparison) ────────────────
TEST_QUERIES = [
    {
        "query": "How does Flash Attention reduce memory complexity?",
        "expected_sources": ["phase3-flash-attention-explained-2.md"],
        "type": "direct"
    },
    {
        "query": "What is Ring AllReduce and how does it synchronize gradients?",
        "expected_sources": ["phase4-distributed-training-interconnects-2.md"],
        "type": "direct"
    },
    {
        "query": "Why does memory bandwidth limit attention at long sequences?",
        "expected_sources": ["phase3-flash-attention-explained-2.md",
                             "phase1-gpu-architecture-primer.md"],
        "type": "paraphrase"
    },
    {
        "query": "What keeps GPUs from being fully utilized during inference?",
        "expected_sources": ["phase1-gpu-architecture-primer.md",
                             "phase5-inference-infrastructure.md"],
        "type": "paraphrase"
    },
    {
        "query": "How does the KV cache problem connect to the memory hierarchy?",
        "expected_sources": ["phase5-inference-infrastructure.md",
                             "phase1-gpu-architecture-primer.md"],
        "type": "cross-document"
    },
    {
        "query": "What is the relationship between LoRA rank and model quality?",
        "expected_sources": ["phase9-fine-tuning-mental-model.md"],
        "type": "direct"
    },
    {
        "query": "Why did DPO replace PPO-based RLHF in practice?",
        "expected_sources": ["phase10-alignment-rlhf-dpo-reward-modeling.md"],
        "type": "direct"
    },
    {
        "query": "What does LLM-as-judge measure and what are its failure modes?",
        "expected_sources": ["phase11-evals-mental-model.md"],
        "type": "direct"
    },
    {
        "query": "What is the binding constraint in distributed training at scale?",
        "expected_sources": ["phase4-distributed-training-interconnects-2.md"],
        "type": "abstract"
    },
    {
        "query": "Why does quantization improve throughput but not always latency?",
        "expected_sources": ["phase5-inference-infrastructure.md"],
        "type": "abstract"
    },
]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client     = anthropic.Anthropic(api_key=api_key)
    embed_model = SentenceTransformer(EMBED_MODEL)
    reranker   = CrossEncoder(RERANK_MODEL)

    print("Building index...")
    corpus = load_corpus(CORPUS_DIR)
    store, _ = build_index(corpus, embed_model)
    print(f"Index: {len(store)} vectors from {len(corpus)} documents\n")

    all_results = []
    summary = {
        "baseline":      {"hit_rates": [], "top1": []},
        "hyde":          {"hit_rates": [], "top1": []},
        "rerank":        {"hit_rates": [], "top1": []},
        "hyde_rerank":   {"hit_rates": [], "top1": []},
    }

    for i, test in enumerate(TEST_QUERIES):
        query    = test["query"]
        expected = test["expected_sources"]
        qtype    = test["type"]

        print(f"[{i+1}/10] ({qtype}) {query}")

        # Baseline: raw embedding retrieval (replicates Exercise 12.2)
        q_emb = embed_model.encode(query, normalize_embeddings=True)
        base_results = store.search(q_emb, k=TOP_K)
        base_eval = evaluate_retrieval(base_results, expected)

        # HyDE
        hyde_results, hyde_ms, hypothesis = hyde_retrieve(
            query, store, embed_model, client
        )
        hyde_eval = evaluate_retrieval(hyde_results, expected)

        # Reranking
        rerank_results, rerank_ms = rerank_retrieve(
            query, store, embed_model, reranker
        )
        rerank_eval = evaluate_retrieval(rerank_results, expected)

        # HyDE + Reranking
        hyde_rerank_results, hyde_rerank_ms, _ = hyde_rerank_retrieve(
            query, store, embed_model, reranker, client
        )
        hyde_rerank_eval = evaluate_retrieval(hyde_rerank_results, expected)

        print(f"  Hypothesis: {hypothesis[:120]}...")
        print(f"  {'Method':<20} {'Hit Rate':>10} {'Top-1':>8} {'Latency':>10}")
        print(f"  {'Baseline':<20} {base_eval['hit_rate']:>10.0%} "
              f"{'Y' if base_eval['top1_correct'] else 'N':>8}")
        print(f"  {'HyDE':<20} {hyde_eval['hit_rate']:>10.0%} "
              f"{'Y' if hyde_eval['top1_correct'] else 'N':>8} {hyde_ms:>9.0f}ms")
        print(f"  {'Rerank':<20} {rerank_eval['hit_rate']:>10.0%} "
              f"{'Y' if rerank_eval['top1_correct'] else 'N':>8} {rerank_ms:>9.0f}ms")
        print(f"  {'HyDE+Rerank':<20} {hyde_rerank_eval['hit_rate']:>10.0%} "
              f"{'Y' if hyde_rerank_eval['top1_correct'] else 'N':>8} "
              f"{hyde_rerank_ms:>9.0f}ms")
        print()

        for method, ev in [
            ("baseline",    base_eval),
            ("hyde",        hyde_eval),
            ("rerank",      rerank_eval),
            ("hyde_rerank", hyde_rerank_eval),
        ]:
            summary[method]["hit_rates"].append(ev["hit_rate"])
            summary[method]["top1"].append(ev["top1_correct"])

        all_results.append({
            "query": query,
            "type": qtype,
            "expected": expected,
            "hypothesis": hypothesis,
            "baseline":    {**base_eval},
            "hyde":        {**hyde_eval},
            "rerank":      {**rerank_eval},
            "hyde_rerank": {**hyde_rerank_eval},
        })

    print("=" * 60)
    print("IMPROVEMENT SUMMARY vs BASELINE")
    print("=" * 60)
    print(f"{'Method':<20} {'Hit Rate':>10} {'Top-1':>8}")
    print("-" * 40)
    for method in ["baseline", "hyde", "rerank", "hyde_rerank"]:
        hr  = np.mean(summary[method]["hit_rates"])
        t1  = np.mean(summary[method]["top1"])
        print(f"  {method:<18} {hr:>10.0%} {t1:>8.0%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "03_advanced_rag_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    main()