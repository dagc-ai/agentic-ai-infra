"""
Phase 11 -- Exercise 2: LLM-as-judge evaluation
Runs Claude Sonnet as judge against all 150 responses using a 4-dimension rubric.
Output: phase11/data/results/judge_scores.json

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python phase11/scripts/02_judge_responses.py

Rubric version: 1.1
Change from v1.0: tightened technical_accuracy score-2 anchor to explicitly
catch wrong definitions/specs wrapped in otherwise correct explanations.
Calibration finding: judge had systematic +bias on 17/20 responses at v1.0.
"""

import os
import json
import time
from anthropic import Anthropic

# ── Config ────────────────────────────────────────────────────────────────────

RESPONSES_PATH  = "phase11/data/results/raw_responses.json"
OUTPUT_PATH     = "phase11/data/results/judge_scores.json"

JUDGE_MODEL     = "claude-sonnet-4-20250514"
MAX_TOKENS      = 1024

REQUEST_DELAY_SEC = 0.5

# ── Credentials ───────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY='your-key'"
    )

client = Anthropic()

# ── Rubric v1.1 ───────────────────────────────────────────────────────────────

RUBRIC = """
You are evaluating technical AI infrastructure content on four dimensions.
Score each dimension 1-5 using the anchors below.

DIMENSION 1: Technical Accuracy
Does the response state things that are actually true?
1 = Contains specific false claims: wrong numbers, wrong mechanisms, wrong causal relationships
2 = Contains at least one clearly wrong specific claim (wrong spec, wrong definition, wrong formula)
    even if the surrounding explanation is mostly correct. Example: calling NF4 'Neuron Format 4'
    instead of 'Normal Float 4', or stating register file size as '32-64 bytes' instead of ~256KB per SM.
    A wrong definition is a wrong claim regardless of how well the rest of the response reads.
3 = No outright false claims but contains vague or unverifiable specifics that cannot be confirmed
    against a primary source
4 = Accurate throughout with no identifiable false or unverifiable claims
5 = Accurate and precise -- every specific claim could be verified against a primary source

DIMENSION 2: Calibration
Does the response express appropriate confidence relative to what it actually knows?
1 = States fabricated specifics with full confidence -- confident hallucination
2 = Overclaims on uncertain details or understates well-established facts
3 = Mostly well-calibrated with occasional overconfidence on specific details
4 = Consistently confident on established facts, hedges appropriately on details
5 = Perfectly calibrated -- confidence tracks actual epistemic status throughout

DIMENSION 3: Mechanistic Depth
Does the response explain WHY, not just WHAT?
1 = Vague or circular -- "Flash Attention is faster because it is more efficient"
2 = Names the mechanism but does not explain how it works
3 = Explains the mechanism at a surface level without the underlying logic
4 = Explains both the mechanism and the reason it produces the stated outcome
5 = Full mechanistic chain -- a practitioner reading this understands the causal logic completely

DIMENSION 4: Audience Calibration
Is the depth appropriate for a technical practitioner who is not an ML researcher?
1 = Either marketing language with no technical content, or dense notation with no explanation
2 = Skews too far in one direction -- either too shallow or too research-paper-level
3 = Roughly appropriate but with sections that are either too basic or too advanced
4 = Consistently appropriate -- assumes systems thinking, does not assume ML research background
5 = Perfect fit -- concrete, specific, rewards a technically sophisticated but non-specialist reader

IMPORTANT INSTRUCTIONS:
- Do NOT reward length. A concise accurate answer scores the same as a verbose accurate answer.
- Do NOT reward confident delivery. Confident hallucination scores 1 on both accuracy and calibration.
- Do NOT penalize appropriate hedging. Saying "this depends on implementation" when it genuinely
  does is a sign of good calibration, not weakness.
- A response that gets the core definition wrong scores 2 or below on technical_accuracy regardless
  of how well-structured or fluent the rest of the response is.
- Score what is actually in the response, not what you would have written.
"""

JUDGE_PROMPT_TEMPLATE = """
{rubric}

Now evaluate the following response.

QUESTION ASKED:
{prompt}

RESPONSE TO EVALUATE:
{response}

Return your evaluation as valid JSON with this exact structure:
{{
  "scores": {{
    "technical_accuracy": <1-5>,
    "calibration": <1-5>,
    "mechanistic_depth": <1-5>,
    "audience_calibration": <1-5>
  }},
  "mean_score": <float, average of the four scores>,
  "reasoning": {{
    "technical_accuracy": "<one sentence explaining this score>",
    "calibration": "<one sentence explaining this score>",
    "mechanistic_depth": "<one sentence explaining this score>",
    "audience_calibration": "<one sentence explaining this score>"
  }},
  "flag": "<null, or one of: confident_hallucination | vague | too_shallow | too_technical>"
}}

Return only the JSON object. No preamble, no explanation outside the JSON.
"""

# ── Judge a single response ───────────────────────────────────────────────────

def judge_response(prompt_text, response_text, retries=3):
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        rubric=RUBRIC,
        prompt=prompt_text,
        response=response_text
    )

    for attempt in range(retries):
        try:
            result = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            raw = result.content[0].text.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            assert "scores" in parsed
            assert "mean_score" in parsed
            assert "reasoning" in parsed
            required_dims = [
                "technical_accuracy", "calibration",
                "mechanistic_depth", "audience_calibration"
            ]
            for dim in required_dims:
                assert dim in parsed["scores"], f"Missing dimension: {dim}"
                assert dim in parsed["reasoning"], f"Missing reasoning: {dim}"

            scores = parsed["scores"]
            parsed["mean_score"] = round(
                sum(scores.values()) / len(scores), 3
            )

            return parsed

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            print(f"    Parse error attempt {attempt+1}: {e}")
            time.sleep(1)
        except Exception as e:
            print(f"    API error attempt {attempt+1}: {e}")
            time.sleep(2)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(RESPONSES_PATH) as f:
        data = json.load(f)
    responses = data["results"]
    print(f"Loaded {len(responses)} responses from {RESPONSES_PATH}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Rubric version: 1.1")
    print(f"Estimated time: ~{len(responses) * REQUEST_DELAY_SEC / 60:.1f} minutes\n")

    all_scored = []
    failed = []

    for i, r in enumerate(responses):
        variant    = r["variant"]
        prompt_id  = r["prompt_id"]

        print(
            f"[{i+1:03d}/{len(responses)}] {variant:4s} | {prompt_id:12s}",
            end="",
            flush=True
        )

        scores = judge_response(r["prompt"], r["response"])

        if scores is None:
            print(f" -- FAILED")
            failed.append({"prompt_id": prompt_id, "variant": variant})
            all_scored.append({**r, "judge_scores": None})
            continue

        mean = scores["mean_score"]
        flag = scores.get("flag", None)
        flag_str = f" [{flag}]" if flag and flag != "null" else ""
        print(f" -- mean={mean:.2f}{flag_str}")

        all_scored.append({
            **r,
            "judge_scores":    scores["scores"],
            "judge_mean":      scores["mean_score"],
            "judge_reasoning": scores["reasoning"],
            "judge_flag":      scores.get("flag", None),
        })

        time.sleep(REQUEST_DELAY_SEC)

    from collections import defaultdict

    by_variant     = defaultdict(list)
    by_variant_cat = defaultdict(lambda: defaultdict(list))
    flag_counts    = defaultdict(int)

    for r in all_scored:
        if r.get("judge_scores") is None:
            continue
        v    = r["variant"]
        cat  = r["category"]
        mean = r["judge_mean"]
        by_variant[v].append(mean)
        by_variant_cat[v][cat].append(mean)

        flag = r.get("judge_flag")
        if flag and flag != "null":
            flag_counts[f"{v}:{flag}"] += 1

    summary = {}
    for variant, scores_list in sorted(by_variant.items()):
        avg = sum(scores_list) / len(scores_list)
        summary[variant] = {
            "mean_score":  round(avg, 3),
            "n_scored":    len(scores_list),
            "by_category": {
                cat: round(sum(s)/len(s), 3)
                for cat, s in sorted(by_variant_cat[variant].items())
            }
        }

    output = {
        "phase":             "11",
        "rubric_version":    "1.1",
        "judge_model":       JUDGE_MODEL,
        "rubric_dimensions": [
            "technical_accuracy",
            "calibration",
            "mechanistic_depth",
            "audience_calibration"
        ],
        "summary":     summary,
        "flag_counts": dict(flag_counts),
        "failed":      failed,
        "results":     all_scored
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("JUDGE SUMMARY (rubric v1.1)")
    print("="*60)
    for variant, stats in sorted(summary.items()):
        print(f"\n{variant}: mean={stats['mean_score']:.3f} (n={stats['n_scored']})")
        for cat, score in stats["by_category"].items():
            print(f"  {cat:30s} {score:.3f}")

    if flag_counts:
        print("\nFLAG COUNTS:")
        for flag, count in sorted(flag_counts.items()):
            print(f"  {flag}: {count}")

    if failed:
        print(f"\nFAILED: {len(failed)} responses")

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
