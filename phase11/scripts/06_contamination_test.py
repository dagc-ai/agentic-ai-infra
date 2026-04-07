"""
Phase 11 -- Exercise 4: Benchmark contamination test
Takes 10 prompts from the eval set, creates rephrased versions that preserve
meaning but change surface form, runs the judge on both original and rephrased
responses from raw_responses.json, then measures score delta.

If score drops significantly on rephrased vs original, the model is
pattern-matching surface form rather than reasoning from knowledge.

Threshold: >15 percentage point drop on rephrased = contamination signal.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python phase11/scripts/06_contamination_test.py
"""

import os
import json
import time
from anthropic import Anthropic

RESPONSES_PATH = "phase11/data/results/raw_responses.json"
OUTPUT_PATH    = "phase11/data/results/contamination_results.json"
JUDGE_MODEL    = "claude-sonnet-4-20250514"
REQUEST_DELAY  = 0.5

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )

client = Anthropic()

# ── Rephrased prompts ─────────────────────────────────────────────────────────
# Each entry: original prompt_id + rephrased version
# Rephrasing rules: change sentence structure, vocabulary, framing
# Preserve: the exact technical concept being asked about

REPHRASED_PROMPTS = [
    {
        "prompt_id": "gpu_001",
        "original":  "Explain what arithmetic intensity is and why it determines whether a GPU kernel is memory-bandwidth bound or compute bound.",
        "rephrased": "What does the ratio of floating point operations to bytes transferred tell you about a GPU kernel's performance bottleneck, and what term describes this ratio?"
    },
    {
        "prompt_id": "kernel_002",
        "original":  "Why does naive attention have O(N squared) memory complexity and what specific algorithmic change does Flash Attention make to reduce this?",
        "rephrased": "Walk through why storing attention weights for a sequence of length N requires memory that grows quadratically, and describe the key insight that allows Flash Attention to avoid this."
    },
    {
        "prompt_id": "dist_001",
        "original":  "Explain Ring AllReduce step by step -- what each GPU sends, what it receives, and why bus bandwidth utilization approaches 100 percent as the ring size grows.",
        "rephrased": "Describe the collective communication algorithm where GPUs are arranged in a circular topology for gradient aggregation -- what data moves at each step and why the algorithm becomes more bandwidth-efficient at larger scale."
    },
    {
        "prompt_id": "inf_001",
        "original":  "Explain the KV cache -- what it stores, why it grows with sequence length, and what GPU memory constraint it creates at 100K context.",
        "rephrased": "During autoregressive generation, what intermediate tensors must be retained from previous tokens, how does the size of this storage scale with context length, and what does this imply for GPU memory at very long sequences?"
    },
    {
        "prompt_id": "arch_002",
        "original":  "What is Grouped Query Attention, how does it differ from Multi-Head Attention, and what is the direct impact on KV cache size for a 70B model at 100K context?",
        "rephrased": "Describe the attention variant where query heads outnumber key-value heads, contrast it with the standard approach where all head types are equal in number, and quantify what this means for memory usage during long-context inference on a large model."
    },
    {
        "prompt_id": "ft_001",
        "original":  "Explain the LoRA matrix decomposition -- what W plus BA means, why rank r determines the capacity of the update, and why this works for behavioral fine-tuning.",
        "rephrased": "How does Low-Rank Adaptation represent weight updates as a product of two smaller matrices, what controls the expressiveness of this representation, and why is a low-dimensional update sufficient for teaching a model new behaviors?"
    },
    {
        "prompt_id": "ft_005",
        "original":  "Why did DPO largely replace PPO-based RLHF in practice -- what specific complexity and instability problems does DPO eliminate?",
        "rephrased": "What practical engineering problems made reinforcement learning from human feedback difficult to run at scale, and how did Direct Preference Optimization sidestep those problems?"
    },
    {
        "prompt_id": "arch_001",
        "original":  "Explain RoPE -- what problem it solves compared to learned absolute position embeddings and how encoding position as a rotation enables context length extrapolation.",
        "rephrased": "What limitation of fixed learned position tables motivated Rotary Position Embedding, and what property of representing position as a rotation applied to query and key vectors allows the model to generalize to sequence lengths beyond its training distribution?"
    },
    {
        "prompt_id": "inf_002",
        "original":  "What is PagedAttention and how does managing the KV cache like OS virtual memory increase effective batch size?",
        "rephrased": "Describe the memory management technique in vLLM that borrows the concept of non-contiguous page allocation from operating systems -- what problem does it solve and what serving efficiency improvement does it produce?"
    },
    {
        "prompt_id": "agent_002",
        "original":  "What is RAG, what problem does it solve for LLMs with fixed training cutoffs, and what are the two components whose quality determines end-to-end RAG performance?",
        "rephrased": "Describe the pattern of augmenting language model generation with external document retrieval -- what fundamental limitation of parametric models does it address, and what two subsystems must both work well for the overall system to produce accurate answers?"
    },
]

# ── Rubric (v1.1 -- same as judge script) ────────────────────────────────────

RUBRIC = """
You are evaluating technical AI infrastructure content on four dimensions.
Score each dimension 1-5 using the anchors below.

DIMENSION 1: Technical Accuracy
1 = Contains specific false claims: wrong numbers, wrong mechanisms, wrong causal relationships
2 = Contains at least one clearly wrong specific claim (wrong spec, wrong definition, wrong formula)
    even if the surrounding explanation is mostly correct
3 = No outright false claims but contains vague or unverifiable specifics
4 = Accurate throughout with no identifiable false or unverifiable claims
5 = Accurate and precise -- every specific claim could be verified against a primary source

DIMENSION 2: Calibration
1 = States fabricated specifics with full confidence -- confident hallucination
2 = Overclaims on uncertain details or understates well-established facts
3 = Mostly well-calibrated with occasional overconfidence
4 = Consistently confident on established facts, hedges appropriately on details
5 = Perfectly calibrated throughout

DIMENSION 3: Mechanistic Depth
1 = Vague or circular explanation
2 = Names the mechanism but does not explain how it works
3 = Surface-level explanation without underlying logic
4 = Explains both mechanism and why it produces the stated outcome
5 = Full causal chain -- practitioner understands the logic completely

DIMENSION 4: Audience Calibration
1 = Marketing language or impenetrable notation
2 = Too shallow or too research-paper-level
3 = Roughly appropriate with some sections off
4 = Consistently appropriate for technical practitioners
5 = Perfect fit for technically sophisticated non-specialist reader

IMPORTANT: Do NOT reward length or confident delivery. A wrong definition
scores 2 or below on technical_accuracy regardless of surrounding fluency.
"""

JUDGE_PROMPT = """
{rubric}

QUESTION ASKED:
{prompt}

RESPONSE TO EVALUATE:
{response}

Return only valid JSON:
{{
  "scores": {{
    "technical_accuracy": <1-5>,
    "calibration": <1-5>,
    "mechanistic_depth": <1-5>,
    "audience_calibration": <1-5>
  }},
  "mean_score": <float>,
  "reasoning": {{
    "technical_accuracy": "<one sentence>",
    "calibration": "<one sentence>",
    "mechanistic_depth": "<one sentence>",
    "audience_calibration": "<one sentence>"
  }},
  "flag": "<null, or one of: confident_hallucination | vague | too_shallow | too_technical>"
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def judge(prompt_text, response_text, retries=3):
    content = JUDGE_PROMPT.format(
        rubric=RUBRIC,
        prompt=prompt_text,
        response=response_text
    )
    for attempt in range(retries):
        try:
            result = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": content}]
            )
            raw = result.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            scores = parsed["scores"]
            parsed["mean_score"] = round(sum(scores.values()) / 4, 3)
            return parsed
        except Exception as e:
            print(f"    Error attempt {attempt+1}: {e}")
            time.sleep(1)
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing responses -- use DPO variant for consistency
    with open(RESPONSES_PATH) as f:
        data = json.load(f)

    # Build lookup: prompt_id -> response text per variant
    response_lookup = {}
    for r in data["results"]:
        response_lookup[(r["prompt_id"], r["variant"])] = r["response"]

    print(f"Contamination test: {len(REPHRASED_PROMPTS)} prompt pairs")
    print(f"Variant: DPO (highest overall score from Exercise 1)")
    print(f"Estimated time: ~{len(REPHRASED_PROMPTS) * 2 * REQUEST_DELAY / 60:.1f} minutes\n")
    print(f"{'prompt_id':<14} {'orig':>6} {'reph':>6} {'delta':>8} {'signal'}")
    print("-"*55)

    results = []
    for p in REPHRASED_PROMPTS:
        pid      = p["prompt_id"]
        orig_q   = p["original"]
        reph_q   = p["rephrased"]
        response = response_lookup.get((pid, "DPO"), "")

        if not response:
            print(f"{pid:<14} -- no response found, skipping")
            continue

        # Score original prompt + response
        orig_scores = judge(orig_q, response)
        time.sleep(REQUEST_DELAY)

        # Score rephrased prompt + same response
        reph_scores = judge(reph_q, response)
        time.sleep(REQUEST_DELAY)

        if not orig_scores or not reph_scores:
            print(f"{pid:<14} -- judge failed, skipping")
            continue

        orig_mean = orig_scores["mean_score"]
        reph_mean = reph_scores["mean_score"]
        delta     = round(reph_mean - orig_mean, 3)

        # Contamination signal: score drops on rephrased version
        signal = ""
        if delta <= -0.75:
            signal = "CONTAMINATION SIGNAL"
        elif delta <= -0.25:
            signal = "mild drop"
        elif delta >= 0.25:
            signal = "rephrased better"
        else:
            signal = "stable"

        print(f"{pid:<14} {orig_mean:>6.2f} {reph_mean:>6.2f} {delta:>8.3f}  {signal}")

        results.append({
            "prompt_id":       pid,
            "variant":         "DPO",
            "original_prompt": orig_q,
            "rephrased_prompt": reph_q,
            "response":        response,
            "original_scores": orig_scores["scores"],
            "rephrased_scores": reph_scores["scores"],
            "original_mean":   orig_mean,
            "rephrased_mean":  reph_mean,
            "delta":           delta,
            "signal":          signal,
        })

    # Summary
    if results:
        avg_delta = sum(r["delta"] for r in results) / len(results)
        contamination_count = sum(1 for r in results if r["signal"] == "CONTAMINATION SIGNAL")
        stable_count        = sum(1 for r in results if r["signal"] == "stable")

        print("\n" + "="*55)
        print("CONTAMINATION TEST SUMMARY")
        print("="*55)
        print(f"Prompts tested:        {len(results)}")
        print(f"Average delta:         {avg_delta:.3f}")
        print(f"Contamination signals: {contamination_count}/{len(results)}")
        print(f"Stable (within 0.25):  {stable_count}/{len(results)}")

        if contamination_count >= 3:
            print("\nFINDING: Multiple contamination signals detected.")
            print("Model performance is partially driven by surface-form memorization.")
        elif avg_delta < -0.3:
            print("\nFINDING: Consistent mild drop on rephrasing.")
            print("Some surface-form sensitivity present but not strong memorization.")
        else:
            print("\nFINDING: Scores stable across rephrasing.")
            print("No strong contamination signal -- performance reflects reasoning, not recall.")

        output = {
            "phase":                "11",
            "variant_tested":       "DPO",
            "n_tested":             len(results),
            "average_delta":        round(avg_delta, 3),
            "contamination_count":  contamination_count,
            "stable_count":         stable_count,
            "results":              results
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
