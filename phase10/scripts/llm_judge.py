# phase10/scripts/llm_judge.py
# Use Claude Sonnet as judge to score base vs SFT vs DPO responses
# Before running:
#   export ANTHROPIC_API_KEY="your_key_here"

import os
import json
import time
import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

RUBRIC = """
Score the response on three dimensions, each 1-5:

1. technical_accuracy: Are the technical claims correct? Penalize confident wrong answers heavily.
2. conciseness: Does the response get to the point without padding? Penalize excessive repetition.
3. hallucination_avoidance: Does the response avoid stating things it cannot know as fact? Penalize invented numbers and false attributions.

Respond in valid JSON only, no preamble:
{"technical_accuracy": <int>, "conciseness": <int>, "hallucination_avoidance": <int>, "reasoning": "<one sentence>"}
"""

def judge(prompt, response, model_label):
    judge_prompt = f"""You are evaluating an AI assistant's response to a technical question about AI infrastructure.

Question: {prompt}

Response to evaluate ({model_label}):
{response}

{RUBRIC}"""

    result = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    raw = result.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"technical_accuracy": 0, "conciseness": 0, "hallucination_avoidance": 0, "reasoning": f"Parse error: {raw[:100]}"}

# Load comparison results
with open("phase10/data/comparison_results.json") as f:
    results = json.load(f)

scores = []

for i, r in enumerate(results):
    print(f"\nPrompt {i+1}/10: {r['prompt'][:60]}...")
    entry = {"prompt": r["prompt"], "scores": {}}

    for model_label in ["base", "sft", "dpo"]:
        score = judge(r["prompt"], r[model_label], model_label)
        entry["scores"][model_label] = score
        mean = round(sum([score["technical_accuracy"], score["conciseness"], score.get("hallucination_avoidance", score.get("hallucination", 0))]) / 3, 2)
        print(f"  {model_label.upper()}: accuracy={score['technical_accuracy']} conciseness={score['conciseness']} hallucination={score.get('hallucination_avoidance', score.get('hallucination', 0))} mean={mean} | {score['reasoning']}")
        time.sleep(0.5)  # avoid rate limits

    scores.append(entry)

# Aggregate
print("\n=== AGGREGATE SCORES ===")
for model_label in ["base", "sft", "dpo"]:
    acc = sum(s["scores"][model_label]["technical_accuracy"] for s in scores) / len(scores)
    con = sum(s["scores"][model_label]["conciseness"] for s in scores) / len(scores)
    hal = sum(s["scores"][model_label].get("hallucination_avoidance", s["scores"][model_label].get("hallucination", 0)) for s in scores) / len(scores)
    mean = round((acc + con + hal) / 3, 2)
    print(f"{model_label.upper()}: accuracy={acc:.2f} conciseness={con:.2f} hallucination={hal:.2f} mean={mean}")

os.makedirs("phase10/data", exist_ok=True)
with open("phase10/data/judge_scores.json", "w") as f:
    json.dump(scores, f, indent=2)

print("\nScores saved to phase10/data/judge_scores.json")
print("Commit message:")
print("phase10: LLM-as-judge complete | Claude Sonnet scoring base vs SFT vs DPO | 10 prompts x 3 dimensions")
