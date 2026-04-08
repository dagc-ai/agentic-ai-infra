"""
Exercise 12.5 -- RAG Evaluation with RAGAS
Uses RAGAS 0.4.x with OpenAI as the judge LLM.
Runs on Python 3.11 (venv-ragas environment).

Four metrics:
- Faithfulness:      Does the answer only use retrieved context?
- Answer Relevance:  Does the answer address the question?
- Context Precision: Are retrieved chunks relevant to the question?
- Context Recall:    Does context contain enough to answer completely?

Key finding we expect to see: Query 7 (DPO vs PPO) scores high on
faithfulness but low on answer relevance -- the model correctly refused
to hallucinate but the answer is useless. Hit rate from Ex12.2 showed
100%. RAGAS shows the real story.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import anthropic

CORPUS_DIR  = Path("../data/corpus")
RESULTS_DIR = Path("../data/results")
EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE  = 400
CHUNK_OVERLAP = 50
TOP_K = 5

# ── Pipeline (reused from Exercise 12.2) ─────────────────────────────────────
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
        self.embeddings = []
        self.chunks = []

    def add(self, chunk, embedding):
        self.embeddings.append(embedding)
        self.chunks.append(chunk)

    def search(self, query_embedding, k=TOP_K):
        matrix = np.array(self.embeddings)
        scores = matrix @ query_embedding
        top_k = np.argsort(scores)[::-1][:k]
        return [RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
                for i in top_k]

def load_and_index(corpus_dir, embed_model):
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        with open(path) as f:
            content = f.read()
        words = content.split()
        start, idx = 0, 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunks.append(Chunk(
                text=" ".join(words[start:end]),
                source_file=path.name,
                chunk_index=idx,
                word_count=end - start,
            ))
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
            idx += 1

    embeddings = embed_model.encode(
        [c.text for c in chunks],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    store = SimpleVectorStore()
    for chunk, emb in zip(chunks, embeddings):
        store.add(chunk, emb)
    return store, chunks

def retrieve(query, store, embed_model, k=TOP_K):
    q_emb = embed_model.encode(query, normalize_embeddings=True)
    return store.search(q_emb, k=k)

def generate_answer(query, retrieved, client):
    context = "\n\n---\n\n".join(
        f"[Source: {r.chunk.source_file}]\n{r.chunk.text}"
        for r in retrieved
    )
    prompt = f"""You are a technical assistant. Use ONLY the provided context.
If the context does not contain enough information, say so explicitly.

Context:
{context}

Question: {query}

Answer:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    {"query": "How does Flash Attention reduce memory complexity?",
     "ground_truth": "Flash Attention reduces memory complexity from O(N²) to O(N) by using tiling and the online softmax algorithm to avoid materializing the full attention matrix in HBM.",
     "type": "direct"},
    {"query": "What is Ring AllReduce and how does it synchronize gradients?",
     "ground_truth": "Ring AllReduce synchronizes gradients across GPUs in two phases: reduce-scatter distributes partial sums around a ring, then allgather broadcasts the complete sums back to all ranks.",
     "type": "direct"},
    {"query": "Why does memory bandwidth limit attention at long sequences?",
     "ground_truth": "Naive attention writes the full N×N attention matrix to HBM, requiring O(N²) memory bandwidth. At long sequences this saturates memory bandwidth and dominates compute time.",
     "type": "paraphrase"},
    {"query": "What keeps GPUs from being fully utilized during inference?",
     "ground_truth": "During inference at small batch sizes, the arithmetic intensity is very low -- weight matrices are large but only a small amount of arithmetic is performed per byte loaded. This makes inference memory-bandwidth bound and leaves compute units idle.",
     "type": "paraphrase"},
    {"query": "How does the KV cache problem connect to the memory hierarchy?",
     "ground_truth": "The KV cache grows linearly with sequence length and batch size, eventually exceeding GPU HBM capacity. This is the same memory hierarchy constraint from Phase 1 -- data that doesn't fit in fast memory must go to slower memory, degrading performance.",
     "type": "cross-document"},
    {"query": "What is the relationship between LoRA rank and model quality?",
     "ground_truth": "Higher LoRA rank increases trainable parameters and model capacity, improving quality up to a point of diminishing returns. Loss improves continuously with rank but plateaus, making rank a hyperparameter tradeoff between quality and compute cost.",
     "type": "direct"},
    {"query": "Why did DPO replace PPO-based RLHF in practice?",
     "ground_truth": "DPO replaced PPO-based RLHF because it eliminates the reward model and PPO training loop entirely, directly optimizing the policy from preference pairs. This makes alignment training dramatically simpler, more stable, and cheaper.",
     "type": "direct"},
    {"query": "What does LLM-as-judge measure and what are its failure modes?",
     "ground_truth": "LLM-as-judge scores model outputs on dimensions like accuracy, helpfulness, and conciseness. Its failure modes include length bias (preferring longer answers), confidence bias (preferring confident-sounding answers), and self-preference bias.",
     "type": "direct"},
    {"query": "What is the binding constraint in distributed training at scale?",
     "ground_truth": "At scale, the binding constraint in distributed training is interconnect bandwidth -- the speed at which gradients can be synchronized via AllReduce across GPUs and nodes.",
     "type": "abstract"},
    {"query": "Why does quantization improve throughput but not always latency?",
     "ground_truth": "Quantization reduces model size and memory bandwidth requirements, improving throughput by allowing larger batches. But naive INT4 dequantization adds overhead that can hurt latency at small batch sizes unless fused kernels like Marlin are used.",
     "type": "abstract"},
]

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set.")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not set.")

    # Generation: Claude Haiku (same as Exercise 12.2 for consistency)
    gen_client = anthropic.Anthropic(api_key=anthropic_key)

    # Connectivity test -- verify both APIs respond before running full eval
    print("Testing API connectivity...")
    try:
        test_response = gen_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        print(f"  Anthropic: OK ({test_response.content[0].text.strip()})")
    except Exception as e:
        raise ValueError(f"Anthropic API failed: {e}")

    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=openai_key)
        test_response = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say OK"}]
        )
        print(f"  OpenAI: OK ({test_response.choices[0].message.content.strip()})")
    except Exception as e:
        raise ValueError(f"OpenAI API failed: {e}")

    print("Both APIs responding. Proceeding...\n")

    # RAGAS judge: OpenAI GPT-4o-mini (cost-efficient, well-tested with RAGAS)
    ragas_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
    )

    # Embed model for retrieval (same as all previous exercises)
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Building index...")
    store, _ = load_and_index(CORPUS_DIR, embed_model)
    print(f"Index: {len(store.chunks)} vectors from {len(list(CORPUS_DIR.glob('*.md')))} documents\n")

    # ── Run pipeline on all queries, collect RAGAS inputs ────────────────────
    ragas_data = {
        "question":  [],
        "answer":    [],
        "contexts":  [],
        "ground_truth": [],
    }
    query_meta = []

    print("Running retrieval and generation...")
    for i, test in enumerate(TEST_QUERIES):
        query        = test["query"]
        ground_truth = test["ground_truth"]
        qtype        = test["type"]

        retrieved = retrieve(query, store, embed_model)
        answer    = generate_answer(query, retrieved, gen_client)
        contexts  = [r.chunk.text for r in retrieved]

        ragas_data["question"].append(query)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(ground_truth)

        query_meta.append({
            "query":   query,
            "type":    qtype,
            "answer":  answer,
            "sources": [r.chunk.source_file for r in retrieved],
        })

        print(f"  [{i+1}/10] ({qtype}) retrieved {len(retrieved)} chunks")

    # ── Run RAGAS evaluation ──────────────────────────────────────────────────
    print("\nRunning RAGAS evaluation (this makes OpenAI API calls)...")
    dataset = Dataset.from_dict(ragas_data)

    from ragas.run_config import RunConfig

    run_cfg = RunConfig(
        max_workers=1,       # one job at a time -- avoids rate limit bursts
        timeout=120,         # 2 minutes per job
        max_retries=3,
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS SUMMARY")
    print("=" * 60)
    results_df = results.to_pandas()
    print(f"Columns available: {list(results_df.columns)}")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]:
        scores = results_df[metric].dropna()
        print(f"  {metric:<22} mean={scores.mean():.3f}  "
              f"min={scores.min():.3f}  max={scores.max():.3f}")

    print("\n" + "=" * 60)
    print("QUERY 7 -- THE KNOWN FAILURE CASE")
    print("=" * 60)
    q7 = results_df.iloc[6]
    print(f"  Query: {q7['user_input']}")
    print(f"  Answer: {query_meta[6]['answer'][:200]}...")
    print(f"  Faithfulness:      {q7['faithfulness']:.3f}  (expect HIGH -- model refused to hallucinate)")
    print(f"  Answer relevancy:  {q7['answer_relevancy']:.3f}  (expect LOW -- non-answer)")
    print(f"  Context precision: {q7['context_precision']:.3f}")
    print(f"  Context recall:    {q7['context_recall']:.3f}  (expect LOW -- key info missing from chunks)")
    print(f"\n  Hit rate from Exercise 12.2: 100%")
    print(f"  RAGAS tells the real story.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "04_rag_eval_ragas_results.json"
    out_csv  = RESULTS_DIR / "04_rag_eval_ragas_results.csv"

    combined = []
    for i, row in results_df.iterrows():
        combined.append({
            **query_meta[i],
            "faithfulness":      float(row.get("faithfulness", 0)),
            "answer_relevancy":  float(row.get("answer_relevancy", 0)),
            "context_precision": float(row.get("context_precision", 0)),
            "context_recall":    float(row.get("context_recall", 0)),
        })

    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    results_df.to_csv(out_csv, index=False)

    print(f"\nResults saved to {out_json}")
    print(f"CSV saved to {out_csv}")

if __name__ == "__main__":
    main()