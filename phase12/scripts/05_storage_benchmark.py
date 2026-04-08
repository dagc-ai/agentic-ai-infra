"""
Exercise 12.3 -- Storage Benchmark
Chroma vs pgvector/Postgres vs pgvector/CockroachDB

Metrics:
- Ingestion throughput: docs/sec under single writer
- Ingestion throughput: docs/sec under 4 concurrent writers
- Query latency p50/p95/p99 for similarity search
- Query latency p50/p95/p99 for hybrid search (vector + filter)

Hardware: M1 Pro, 16GB unified memory
Corpus: Phase 1-11 notes (106 chunks, 768-dim MPNet embeddings)

Why this benchmark matters:
- Chroma has no consistency guarantees under concurrent writes
- pgvector/Postgres handles concurrent writes with ACID guarantees
- pgvector/CockroachDB adds distributed consensus on top
- The query that's impossible in Chroma (vector + SQL predicate
  in one transaction) is trivial in pgvector
"""

import os
import time
import json
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer

CORPUS_DIR  = Path("../data/corpus")
RESULTS_DIR = Path("../data/results")
EMBED_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE  = 400
CHUNK_OVERLAP = 50
VECTOR_DIM  = 768
N_QUERIES   = 50   # queries for latency benchmark
N_WARMUP    = 5    # warmup queries before measuring

# ── Chunk loading (reused from Exercise 12.2) ─────────────────────────────────
@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int

def load_chunks(corpus_dir: Path) -> List[Chunk]:
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
            ))
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
            idx += 1
    return chunks

def embed_chunks(chunks: List[Chunk], model: SentenceTransformer):
    texts = [c.text for c in chunks]
    return model.encode(texts, normalize_embeddings=True,
                        batch_size=32, show_progress_bar=True)

# ── Latency stats ─────────────────────────────────────────────────────────────
def percentiles(latencies_ms: List[float]) -> Dict:
    a = np.array(latencies_ms)
    return {
        "p50":  float(np.percentile(a, 50)),
        "p95":  float(np.percentile(a, 95)),
        "p99":  float(np.percentile(a, 99)),
        "mean": float(np.mean(a)),
    }

# ── Chroma backend ────────────────────────────────────────────────────────────
class ChromaBackend:
    def __init__(self):
        import chromadb
        self.client = chromadb.Client()
        self.collection = None

    def setup(self):
        try:
            self.client.delete_collection("ragbench")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="ragbench",
            metadata={"hnsw:space": "cosine"}
        )

    def insert_batch(self, chunks: List[Chunk], embeddings: np.ndarray):
        self.collection.add(
            ids=[f"{c.source_file}_{c.chunk_index}" for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source_file,
                        "chunk_index": c.chunk_index} for c in chunks],
        )

    def query_similarity(self, query_embedding: np.ndarray, k: int = 5) -> float:
        start = time.perf_counter()
        self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
        )
        return (time.perf_counter() - start) * 1000

    def query_hybrid(self, query_embedding: np.ndarray,
                     source_filter: str, k: int = 5) -> float:
        """
        Chroma supports metadata filters but they're applied POST-retrieval
        on the client side, not as a true SQL predicate in the index scan.
        This is the fundamental limitation vs pgvector.
        """
        start = time.perf_counter()
        self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where={"source": source_filter},
        )
        return (time.perf_counter() - start) * 1000

    def teardown(self):
        try:
            self.client.delete_collection("ragbench")
        except Exception:
            pass

# ── Postgres pgvector backend ──────────────────────────────────────────────────
class PostgresBackend:
    def __init__(self, dsn: str, backend_name: str = "postgres"):
        self.dsn = dsn
        self.name = backend_name

    def _conn(self):
        import psycopg2
        return psycopg2.connect(self.dsn)

    def setup(self):
        import psycopg2
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS ragbench_docs")
            cur.execute(f"""
                CREATE TABLE ragbench_docs (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({VECTOR_DIM})
                )
            """)
            cur.execute("""
                CREATE INDEX ON ragbench_docs
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10)
            """)
        conn.commit()
        conn.close()

    def insert_batch(self, chunks: List[Chunk], embeddings: np.ndarray):
        from psycopg2.extras import execute_values
        conn = self._conn()
        with conn.cursor() as cur:
            data = [
                (c.source_file, c.chunk_index, c.text,
                 embeddings[i].tolist())
                for i, c in enumerate(chunks)
            ]
            execute_values(cur, """
                INSERT INTO ragbench_docs
                (source_file, chunk_index, content, embedding)
                VALUES %s
            """, data, template="(%s, %s, %s, %s::vector)")
        conn.commit()
        conn.close()

    def query_similarity(self, query_embedding: np.ndarray, k: int = 5) -> float:
        conn = self._conn()
        start = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_file, chunk_index,
                       embedding <=> %s::vector AS distance
                FROM ragbench_docs
                ORDER BY distance
                LIMIT %s
            """, (query_embedding.tolist(), k))
            cur.fetchall()
        elapsed = (time.perf_counter() - start) * 1000
        conn.close()
        return elapsed

    def query_hybrid(self, query_embedding: np.ndarray,
                     source_filter: str, k: int = 5) -> float:
        """
        True hybrid query: vector similarity + SQL predicate
        in a single transaction. This is the query Chroma cannot do.
        """
        conn = self._conn()
        start = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT source_file, chunk_index,
                       embedding <=> %s::vector AS distance
                FROM ragbench_docs
                WHERE source_file = %s
                ORDER BY distance
                LIMIT %s
            """, (query_embedding.tolist(), source_filter, k))
            cur.fetchall()
        elapsed = (time.perf_counter() - start) * 1000
        conn.close()
        return elapsed

    def teardown(self):
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ragbench_docs")
        conn.commit()
        conn.close()

# ── CockroachDB backend (inherits Postgres) ───────────────────────────────────
class CockroachBackend(PostgresBackend):
    def setup(self):
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS ragbench_docs")
            cur.execute(f"""
                CREATE TABLE ragbench_docs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_file STRING NOT NULL,
                    chunk_index INT NOT NULL,
                    content STRING NOT NULL,
                    embedding VECTOR({VECTOR_DIM})
                )
            """)
        conn.commit()
        conn.close()

# ── Benchmark functions ───────────────────────────────────────────────────────
def benchmark_single_writer(backend, chunks, embeddings):
    """Measure ingestion throughput under single writer."""
    backend.setup()
    start = time.perf_counter()
    backend.insert_batch(chunks, embeddings)
    elapsed = time.perf_counter() - start
    docs_per_sec = len(chunks) / elapsed
    return docs_per_sec, elapsed

def benchmark_concurrent_writers(backend, chunks, embeddings, n_workers=4):
    """
    Measure ingestion throughput under 4 concurrent writers.
    Each worker inserts a quarter of the corpus simultaneously.
    This is where Chroma shows its consistency limitations.
    """
    backend.setup()

    chunk_size = len(chunks) // n_workers
    results = {}
    errors  = {}

    def worker(worker_id, chunk_slice, emb_slice):
        try:
            start = time.perf_counter()
            backend.insert_batch(chunk_slice, emb_slice)
            results[worker_id] = time.perf_counter() - start
        except Exception as e:
            errors[worker_id] = str(e)

    threads = []
    wall_start = time.perf_counter()
    for i in range(n_workers):
        start_idx = i * chunk_size
        end_idx   = start_idx + chunk_size if i < n_workers - 1 else len(chunks)
        t = threading.Thread(
            target=worker,
            args=(i, chunks[start_idx:end_idx], embeddings[start_idx:end_idx])
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall_elapsed = time.perf_counter() - wall_start
    docs_per_sec = len(chunks) / wall_elapsed

    return docs_per_sec, wall_elapsed, errors

def benchmark_query_latency(backend, query_embeddings, source_files):
    """Measure p50/p95/p99 for similarity and hybrid queries."""
    sim_latencies    = []
    hybrid_latencies = []

    # Warmup
    for i in range(N_WARMUP):
        backend.query_similarity(query_embeddings[i])

    # Similarity search
    for i in range(N_QUERIES):
        lat = backend.query_similarity(query_embeddings[i % len(query_embeddings)])
        sim_latencies.append(lat)

    # Hybrid search (vector + source file filter)
    for i in range(N_QUERIES):
        source = source_files[i % len(source_files)]
        lat = backend.query_hybrid(
            query_embeddings[i % len(query_embeddings)], source
        )
        hybrid_latencies.append(lat)

    return {
        "similarity": percentiles(sim_latencies),
        "hybrid":     percentiles(hybrid_latencies),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading corpus and computing embeddings...")
    model  = SentenceTransformer(EMBED_MODEL)
    chunks = load_chunks(CORPUS_DIR)
    embeddings = embed_chunks(chunks, model)
    print(f"Corpus: {len(chunks)} chunks, {embeddings.shape[1]}-dim embeddings")

    # Query embeddings: embed the chunk texts themselves as proxy queries
    print("\nPreparing query embeddings...")
    query_texts = [c.text[:100] for c in chunks[:N_QUERIES + N_WARMUP]]
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True, batch_size=32
    )
    source_files = list(set(c.source_file for c in chunks))

    # ── Backend configs ────────────────────────────────────────────────────────
    pg_dsn   = "host=localhost port=5432 dbname=ragbench user=" + os.environ.get("USER", "danielguerrero")
    crdb_dsn = "host=localhost port=26257 dbname=ragbench user=root sslmode=disable"

    backends = [
        ("Chroma",       ChromaBackend()),
        ("pgvector/PG",  PostgresBackend(pg_dsn)),
        ("pgvector/CRDB", CockroachBackend(crdb_dsn, "cockroachdb")),
    ]

    all_results = {}

    for name, backend in backends:
        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {name}")
        print('='*60)

        try:
            # Single writer
            print(f"\n[1/3] Single writer ingestion...")
            sw_tps, sw_elapsed = benchmark_single_writer(
                backend, chunks, embeddings
            )
            print(f"  {sw_tps:.1f} docs/sec ({sw_elapsed:.2f}s for {len(chunks)} chunks)")

            # Query latency (after single writer load)
            print(f"\n[2/3] Query latency ({N_QUERIES} queries each)...")
            latencies = benchmark_query_latency(
                backend, query_embeddings, source_files
            )
            sim = latencies["similarity"]
            hyb = latencies["hybrid"]
            print(f"  Similarity  p50={sim['p50']:.1f}ms  p95={sim['p95']:.1f}ms  p99={sim['p99']:.1f}ms")
            print(f"  Hybrid      p50={hyb['p50']:.1f}ms  p95={hyb['p95']:.1f}ms  p99={hyb['p99']:.1f}ms")

            # Concurrent writers
            print(f"\n[3/3] Concurrent writer ingestion (4 workers)...")
            cw_tps, cw_elapsed, errors = benchmark_concurrent_writers(
                backend, chunks, embeddings, n_workers=4
            )
            if errors:
                print(f"  ERRORS: {errors}")
            print(f"  {cw_tps:.1f} docs/sec ({cw_elapsed:.2f}s wall clock)")
            print(f"  Speedup vs single writer: {cw_tps/sw_tps:.2f}x")

            all_results[name] = {
                "single_writer_docs_per_sec":     sw_tps,
                "single_writer_elapsed_s":        sw_elapsed,
                "concurrent_writer_docs_per_sec": cw_tps,
                "concurrent_writer_elapsed_s":    cw_elapsed,
                "concurrent_writer_errors":       errors,
                "query_latency":                  latencies,
            }

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[name] = {"error": str(e)}
        finally:
            try:
                backend.teardown()
            except Exception:
                pass

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print('='*60)
    print(f"\n{'Backend':<20} {'SW (doc/s)':>12} {'CW (doc/s)':>12} "
          f"{'Sim p50':>10} {'Sim p99':>10} {'Hyb p50':>10} {'Hyb p99':>10}")
    print("-" * 90)

    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:<18} ERROR: {r['error']}")
            continue
        sim = r["query_latency"]["similarity"]
        hyb = r["query_latency"]["hybrid"]
        print(f"  {name:<18} "
              f"{r['single_writer_docs_per_sec']:>12.1f} "
              f"{r['concurrent_writer_docs_per_sec']:>12.1f} "
              f"{sim['p50']:>10.1f} "
              f"{sim['p99']:>10.1f} "
              f"{hyb['p50']:>10.1f} "
              f"{hyb['p99']:>10.1f}")

    print("\nKey finding: Chroma concurrent write errors (if any) confirm")
    print("lack of consistency guarantees under concurrent load.")
    print("pgvector hybrid query runs in a single transaction --")
    print("impossible in Chroma without application-side joining.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "05_storage_benchmark_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")

if __name__ == "__main__":
    main()