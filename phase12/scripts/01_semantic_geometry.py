"""
Exercise 12.1 — Embeddings From First Principles
Semantic geometry experiment using the Phase 1-11 notes corpus.

What we're testing:
- Does the embedding model capture the semantic relationships we expect?
- Where does it get it right? Where does it fail?
- What does this tell us about when retrieval will and won't work?
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Sentence sets ────────────────────────────────────────────────────────────
# Draw directly from your corpus. You know what should be similar.
# Group A: Flash Attention / memory hierarchy -- should cluster tightly
# Group B: Distributed training -- should cluster separately from A
# Group C: Alignment -- should cluster separately from A and B
# Group D: Adversarial pairs -- same words, different meaning
# Group E: Paraphrase pairs -- different words, same meaning

sentences = {
    # Group A: Flash Attention / memory hierarchy
    "FA1": "Flash Attention tiles K and V into SRAM to avoid writing the N×N attention matrix to HBM",
    "FA2": "By computing softmax incrementally with a running max and sum, Flash Attention achieves O(N) memory complexity",
    "FA3": "The tiled matmul loads data from HBM into shared memory once and amortizes the cost across many operations",
    "FA4": "Memory bandwidth is the binding constraint for attention at long sequence lengths",

    # Group B: Distributed training
    "DT1": "Ring AllReduce distributes gradient synchronization across GPUs by sending chunks in a ring topology",
    "DT2": "AllReduce is a synchronization barrier -- the slowest rank determines when all ranks proceed",
    "DT3": "Model FLOP Utilization measures actual useful compute against theoretical peak, accounting for communication overhead",
    "DT4": "NVLink provides 600 GB/s bidirectional bandwidth between GPUs within a node",

    # Group C: Alignment
    "AL1": "DPO eliminates the reward model by rearranging the RLHF objective to directly update the policy from preference pairs",
    "AL2": "The Bradley-Terry preference model trains a reward model to predict which of two responses a human would prefer",
    "AL3": "SFT on confident domain-specific examples can amplify hallucination confidence rather than reducing it",
    "AL4": "Constitutional AI uses AI-generated critique and revision loops to replace or augment human labelers",

    # Group D: Adversarial -- same technical term, different context
    # These should NOT be similar even though they share vocabulary
    "ADV1": "Attention is all you need -- the original transformer paper title",
    "ADV2": "Flash Attention reduces the memory complexity of the attention mechanism from O(N²) to O(N)",
    "ADV3": "Paying attention to the details of the hardware memory hierarchy is what separates fast kernels from slow ones",

    # Group E: Paraphrase pairs -- same meaning, different vocabulary
    # This is the hard test. Can the model bridge different phrasings?
    "PARA1": "HBM round trips are the bottleneck in naive attention",
    "PARA2": "Memory access overhead dominates the cost of standard self-attention",
    "PARA3": "The GPU spends most of its time waiting for data from off-chip memory during attention computation",
}

def compute_similarity_matrix(model_name: str, sentences: dict) -> tuple:
    """Embed all sentences and return cosine similarity matrix."""
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)

    keys = list(sentences.keys())
    texts = [sentences[k] for k in keys]

    print(f"Embedding {len(texts)} sentences...")
    embeddings = model.encode(texts, normalize_embeddings=True)
    # normalize_embeddings=True means cosine sim = dot product
    sim_matrix = np.dot(embeddings, embeddings.T)

    return keys, embeddings, sim_matrix

def print_similarity_analysis(keys, sim_matrix, sentences):
    """Print the similarity analysis in a readable format."""
    n = len(keys)

    print("\n" + "="*70)
    print("PAIRWISE SIMILARITY ANALYSIS")
    print("="*70)

    # Print within-group similarities
    groups = {
        "Flash Attention / Memory": ["FA1", "FA2", "FA3", "FA4"],
        "Distributed Training":     ["DT1", "DT2", "DT3", "DT4"],
        "Alignment":                ["AL1", "AL2", "AL3", "AL4"],
        "Adversarial (same term, different meaning)": ["ADV1", "ADV2", "ADV3"],
        "Paraphrase (different words, same meaning)": ["PARA1", "PARA2", "PARA3"],
    }

    for group_name, group_keys in groups.items():
        print(f"\n[{group_name}]")
        indices = [keys.index(k) for k in group_keys if k in keys]
        sims = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                sim = sim_matrix[indices[i], indices[j]]
                sims.append(sim)
                ki, kj = keys[indices[i]], keys[indices[j]]
                print(f"  {ki} vs {kj}: {sim:.3f}")
        if sims:
            print(f"  --> Group mean similarity: {np.mean(sims):.3f}")

    # Print cross-group similarities (should be low)
    print("\n[Cross-group: Flash Attention vs Distributed Training]")
    fa_idx  = [keys.index(k) for k in ["FA1", "FA2", "FA3", "FA4"]]
    dt_idx  = [keys.index(k) for k in ["DT1", "DT2", "DT3", "DT4"]]
    cross = [sim_matrix[i, j] for i in fa_idx for j in dt_idx]
    print(f"  Mean cross-group similarity: {np.mean(cross):.3f}")

    print("\n[Cross-group: Flash Attention vs Alignment]")
    al_idx = [keys.index(k) for k in ["AL1", "AL2", "AL3", "AL4"]]
    cross = [sim_matrix[i, j] for i in fa_idx for j in al_idx]
    print(f"  Mean cross-group similarity: {np.mean(cross):.3f}")

    # The diagnostic questions
    print("\n" + "="*70)
    print("DIAGNOSTIC QUESTIONS")
    print("="*70)

    adv1_idx  = keys.index("ADV1")
    adv2_idx  = keys.index("ADV2")
    adv3_idx  = keys.index("ADV3")
    para1_idx = keys.index("PARA1")
    para2_idx = keys.index("PARA2")
    para3_idx = keys.index("PARA3")

    print(f"\nQ1: Does 'Attention is all you need' (paper title) cluster with Flash Attention?")
    print(f"  ADV1 vs ADV2: {sim_matrix[adv1_idx, adv2_idx]:.3f}  (want: LOW)")
    print(f"  ADV1 vs ADV3: {sim_matrix[adv1_idx, adv3_idx]:.3f}  (want: LOW)")
    print(f"  ADV2 vs ADV3: {sim_matrix[adv2_idx, adv3_idx]:.3f}  (want: HIGH)")

    print(f"\nQ2: Can the model bridge paraphrase -- 'HBM round trips' vs 'memory access overhead'?")
    print(f"  PARA1 vs PARA2: {sim_matrix[para1_idx, para2_idx]:.3f}  (want: HIGH)")
    print(f"  PARA1 vs PARA3: {sim_matrix[para1_idx, para3_idx]:.3f}  (want: HIGH)")
    print(f"  PARA2 vs PARA3: {sim_matrix[para2_idx, para3_idx]:.3f}  (want: HIGH)")

    print(f"\nQ3: Does Flash Attention prose (FA1-FA4) cluster more tightly than cross-group?")
    fa_sims = [sim_matrix[fa_idx[i], fa_idx[j]]
               for i in range(len(fa_idx)) for j in range(i+1, len(fa_idx))]
    print(f"  Within FA group mean: {np.mean(fa_sims):.3f}")
    cross_fa_dt = [sim_matrix[i, j] for i in fa_idx for j in dt_idx]
    print(f"  FA vs DT cross mean:  {np.mean(cross_fa_dt):.3f}")
    print(f"  Separation ratio: {np.mean(fa_sims)/np.mean(cross_fa_dt):.1f}x")

if __name__ == "__main__":
    # Run with two models -- compare quality
    models = [
        "all-MiniLM-L6-v2",    # 22M params, 384 dims -- fast baseline
        "all-mpnet-base-v2",    # 110M params, 768 dims -- better quality
    ]

    results = {}
    for model_name in models:
        keys, embeddings, sim_matrix = compute_similarity_matrix(model_name, sentences)
        print_similarity_analysis(keys, sim_matrix, sentences)
        results[model_name] = {
            "keys": keys,
            "embeddings": embeddings,
            "sim_matrix": sim_matrix,
        }

    # Model comparison summary
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<45} {'MiniLM':>8} {'MPNet':>8}")
    print("-"*65)

    for label, k1, k2, direction in [
        ("FA group mean (want HIGH)",         "FA1", "FA2", "high"),
        ("ADV1 vs ADV2 -- paper title (want LOW)", "ADV1", "ADV2", "low"),
        ("PARA1 vs PARA2 -- paraphrase (want HIGH)", "PARA1", "PARA2", "high"),
        ("FA1 vs DT1 -- cross-group (want LOW)", "FA1", "DT1", "low"),
    ]:
        row = f"  {label:<43}"
        for model_name in models:
            r = results[model_name]
            i1 = r["keys"].index(k1)
            i2 = r["keys"].index(k2)
            val = r["sim_matrix"][i1, i2]
            row += f" {val:>8.3f}"
        print(row)

    print("\nRecord these numbers in phase12/notes/embeddings-analysis.md")
    print("Include: which model handles paraphrase better, and what this")
    print("tells you about chunking strategy for the RAG pipeline.")