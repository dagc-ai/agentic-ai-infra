"""
Exercise 12.2 — RAG From Scratch
No frameworks. Every step explicit.

Pipeline:
  1. Ingest markdown files from corpus/
  2. Chunk with fixed-size + overlap strategy
  3. Embed chunks with all-mpnet-base-v2
  4. Store in SimpleVectorStore (in-memory, numpy)
  5. Query: embed question, retrieve top-K chunks
  6. Generate: send retrieved context to Claude
  7. Evaluate: run 10 test queries, record retrieval quality

What this teaches: every RAG framework is this pipeline with abstractions.
Understanding each step makes every framework decision legible.
"""

import os
import json
import time
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import anthropic

# ── Configuration ────────────────────────────────────────────────────────────
CORPUS_DIR   = Path("../data/corpus")
RESULTS_DIR  = Path("../data/results")
MODEL_NAME   = "all-mpnet-base-v2"
CHUNK_SIZE   = 400    # tokens (approximate -- we use words as proxy)
CHUNK_OVERLAP = 50    # words of overlap between adjacent chunks
TOP_K        = 5      # chunks to retrieve per query

# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """One retrievable unit of text."""
    text: str
    source_file: str
    chunk_index: int
    word_count: int

@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float

@dataclass
class RAGResult:
    """Full pipeline result for one query."""
    query: str
    retrieved: List[RetrievalResult]
    answer: str
    retrieval_ms: float
    generation_ms: float

# ── Step 1: Ingestion ────────────────────────────────────────────────────────
def load_corpus(corpus_dir: Path) -> Dict[str, str]:
    """
    Load all markdown files from corpus directory.
    Returns dict of {filename: content}.
    """
    corpus = {}
    for path in sorted(corpus_dir.glob("*.md")):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        corpus[path.name] = content
        print(f"  Loaded: {path.name} ({len(content.split()):,} words)")
    return corpus

# ── Step 2: Chunking ─────────────────────────────────────────────────────────
def chunk_fixed(
    text: str,
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Chunk]:
    """
    Fixed-size chunking with overlap.

    Why overlap? A chunk boundary can split a sentence mid-thought.
    The overlapping words carry forward just enough context that
    neither chunk loses the connecting idea entirely.

    Why words not characters? Character splits are arbitrary.
    Word splits at least preserve token integrity.

    Limitation: this ignores semantic boundaries entirely.
    A paragraph about Ring AllReduce and a paragraph about
    KV cache can land in the same chunk if they're adjacent.
    That averaged embedding retrieves poorly for both.
    Phase 12.4 (advanced RAG) addresses this with semantic chunking.
    """
    words = text.split()
    chunks = []
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(Chunk(
            text=chunk_text,
            source_file=source_file,
            chunk_index=idx,
            word_count=len(chunk_words),
        ))

        if end == len(words):
            break

        start += chunk_size - overlap
        idx += 1

    return chunks

def chunk_corpus(corpus: Dict[str, str]) -> List[Chunk]:
    """Chunk all documents in the corpus."""
    all_chunks = []
    for filename, content in corpus.items():
        chunks = chunk_fixed(content, filename)
        all_chunks.extend(chunks)
        print(f"  {filename}: {len(chunks)} chunks")
    return all_chunks

# ── Step 3 & 4: Embedding + Vector Store ─────────────────────────────────────
class SimpleVectorStore:
    """
    A vector store in ~50 lines.
    This is what Chroma, Pinecone, and Qdrant are doing underneath
    their abstractions -- minus the persistence, indexing optimizations,
    and concurrent access handling we'll benchmark in Exercise 12.3.
    """

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
        scores = matrix @ query_embedding  # dot product of normalized vectors = cosine sim

        top_k_indices = np.argsort(scores)[::-1][:k]

        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_k_indices
        ]

    def __len__(self):
        return len(self.chunks)

def build_index(chunks: List[Chunk], model: SentenceTransformer) -> SimpleVectorStore:
    """Embed all chunks and load into vector store."""
    store = SimpleVectorStore()

    texts = [c.text for c in chunks]
    print(f"\nEmbedding {len(texts)} chunks (this takes ~30-60s on M1)...")

    start = time.time()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    elapsed = time.time() - start
    print(f"Embedding complete: {elapsed:.1f}s ({len(texts)/elapsed:.0f} chunks/sec)")

    for chunk, embedding in zip(chunks, embeddings):
        store.add(chunk, embedding)

    return store

# ── Step 5: Retrieval ─────────────────────────────────────────────────────────
def retrieve(
    query: str,
    store: SimpleVectorStore,
    model: SentenceTransformer,
    k: int = TOP_K
) -> Tuple[List[RetrievalResult], float]:
    """Embed query and retrieve top-K chunks. Returns results and latency."""
    start = time.time()
    query_embedding = model.encode(query, normalize_embeddings=True)
    results = store.search(query_embedding, k=k)
    latency_ms = (time.time() - start) * 1000
    return results, latency_ms

# ── Step 6: Generation ────────────────────────────────────────────────────────
def generate(
    query: str,
    retrieved: List[RetrievalResult],
    client: anthropic.Anthropic
) -> Tuple[str, float]:
    """
    Assemble context from retrieved chunks and generate answer.
    Returns answer text and latency.
    """
    context_parts = []
    for i, result in enumerate(retrieved):
        context_parts.append(
            f"[Source {i+1}: {result.chunk.source_file}, "
            f"chunk {result.chunk.chunk_index}, "
            f"similarity {result.score:.3f}]\n{result.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a technical assistant answering questions about AI infrastructure.
Use ONLY the provided context to answer the question.
If the context does not contain enough information, say so explicitly.
Do not use knowledge outside the provided context.

Context:
{context}

Question: {query}

Answer:"""

    start = time.time()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.time() - start) * 1000
    answer = response.content[0].text

    return answer, latency_ms

# ── Step 7: Test Queries ──────────────────────────────────────────────────────
TEST_QUERIES = [
    # Direct vocabulary match -- should retrieve easily
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
    # Paraphrase -- different vocabulary, same concept
    {
        "query": "Why does memory bandwidth limit attention at long sequences?",
        "expected_sources": ["phase3-flash-attention-explained-2.md", "phase1-gpu-architecture-primer.md"],
        "type": "paraphrase"
    },
    {
        "query": "What keeps GPUs from being fully utilized during inference?",
        "expected_sources": ["phase1-gpu-architecture-primer.md", "phase5-inference-infrastructure.md"],
        "type": "paraphrase"
    },
    # Cross-document -- answer requires synthesizing across files
    {
        "query": "How does the KV cache problem connect to the memory hierarchy?",
        "expected_sources": ["phase5-inference-infrastructure.md", "phase1-gpu-architecture-primer.md"],
        "type": "cross-document"
    },
    {
        "query": "What is the relationship between LoRA rank and model quality?",
        "expected_sources": ["phase9-fine-tuning-mental-model.md"],
        "type": "direct"
    },
    # Phase 10/11 specific
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
    # Hard: abstract concept, no obvious keyword match
    {
        "query": "What is the binding constraint in distributed training at scale?",
        "expected_sources": ["phase4-distributed-training-interconnects-2.md", "full-stack-view.md"],
        "type": "abstract"
    },
    {
        "query": "Why does quantization improve throughput but not always latency?",
        "expected_sources": ["phase5-inference-infrastructure.md"],
        "type": "abstract"
    },
]

def evaluate_retrieval(results: List[RetrievalResult], expected_sources: List[str]) -> Dict:
    """
    Check whether expected sources appear in retrieved results.
    This is a simplified precision check -- formal RAGAS metrics come in Exercise 12.5.
    """
    retrieved_sources = [r.chunk.source_file for r in results]
    hits = sum(1 for s in expected_sources if s in retrieved_sources)

    return {
        "expected": expected_sources,
        "retrieved_sources": retrieved_sources,
        "hits": hits,
        "hit_rate": hits / len(expected_sources) if expected_sources else 0,
        "top1_correct": retrieved_sources[0] in expected_sources if retrieved_sources else False,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=your-key")

    client = anthropic.Anthropic(api_key=api_key)
    model  = SentenceTransformer(MODEL_NAME)

    print("=" * 60)
    print("STEP 1-2: INGESTION AND CHUNKING")
    print("=" * 60)
    corpus = load_corpus(CORPUS_DIR)
    print(f"\nTotal documents: {len(corpus)}")

    chunks = chunk_corpus(corpus)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Avg chunk size: {np.mean([c.word_count for c in chunks]):.0f} words")

    print("\n" + "=" * 60)
    print("STEP 3-4: EMBEDDING AND INDEXING")
    print("=" * 60)
    store = build_index(chunks, model)
    print(f"Index built: {len(store)} vectors")

    print("\n" + "=" * 60)
    print("STEP 5-7: RETRIEVAL AND GENERATION EVALUATION")
    print("=" * 60)

    all_results = []
    hit_rates   = []
    top1_correct = []

    for i, test in enumerate(TEST_QUERIES):
        query    = test["query"]
        expected = test["expected_sources"]
        qtype    = test["type"]

        print(f"\n[Query {i+1}/{len(TEST_QUERIES)}] ({qtype})")
        print(f"  Q: {query}")

        retrieved, ret_ms = retrieve(query, store, model)
        eval_metrics = evaluate_retrieval(retrieved, expected)
        answer, gen_ms = generate(query, retrieved, client)

        hit_rates.append(eval_metrics["hit_rate"])
        top1_correct.append(eval_metrics["top1_correct"])

        print(f"  Retrieved: {[r.chunk.source_file for r in retrieved]}")
        print(f"  Expected:  {expected}")
        print(f"  Hit rate:  {eval_metrics['hit_rate']:.0%} | Top-1 correct: {eval_metrics['top1_correct']}")
        print(f"  Scores:    {[f'{r.score:.3f}' for r in retrieved]}")
        print(f"  Latency:   retrieval={ret_ms:.0f}ms, generation={gen_ms:.0f}ms")
        print(f"  Answer:    {answer[:200]}...")

        all_results.append({
            "query": query,
            "type": qtype,
            "expected_sources": expected,
            "retrieved_sources": eval_metrics["retrieved_sources"],
            "retrieval_scores":  [r.score for r in retrieved],
            "hit_rate":          eval_metrics["hit_rate"],
            "top1_correct":      eval_metrics["top1_correct"],
            "answer":            answer,
            "retrieval_ms":      ret_ms,
            "generation_ms":     gen_ms,
        })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Queries run:         {len(TEST_QUERIES)}")
    print(f"Mean hit rate:       {np.mean(hit_rates):.0%}")
    print(f"Top-1 accuracy:      {np.mean(top1_correct):.0%}")
    print(f"By query type:")

    for qtype in ["direct", "paraphrase", "cross-document", "abstract"]:
        type_results = [r for r in all_results if r["type"] == qtype]
        if type_results:
            mean_hit = np.mean([r["hit_rate"] for r in type_results])
            print(f"  {qtype:<16}: {mean_hit:.0%} hit rate ({len(type_results)} queries)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "02_rag_from_scratch_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")
    print("\nRecord summary numbers in phase12/notes/phase12-embeddings-analysis.md")

if __name__ == "__main__":
    main()