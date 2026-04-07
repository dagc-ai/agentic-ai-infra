"""
Phase 11 -- Score analysis: per-dimension breakdown across variants
Output: printed table + phase11/data/results/score_analysis.json
"""

import json
from collections import defaultdict

with open("phase11/data/results/judge_scores.json") as f:
    data = json.load(f)

results = [r for r in data["results"] if r.get("judge_scores")]

dims = [
    "technical_accuracy",
    "calibration", 
    "mechanistic_depth",
    "audience_calibration"
]

# Per-dimension per-variant averages
by_variant_dim = defaultdict(lambda: defaultdict(list))
for r in results:
    v = r["variant"]
    for dim in dims:
        by_variant_dim[v][dim].append(r["judge_scores"][dim])

print("\nPER-DIMENSION SCORES BY VARIANT")
print(f"{'Dimension':<28} {'BASE':>6} {'SFT':>6} {'DPO':>6}")
print("-" * 50)
for dim in dims:
    scores = {
        v: sum(by_variant_dim[v][dim]) / len(by_variant_dim[v][dim])
        for v in ["BASE", "SFT", "DPO"]
    }
    print(f"{dim:<28} {scores['BASE']:>6.3f} {scores['SFT']:>6.3f} {scores['DPO']:>6.3f}")

# Flag rate per variant
print("\nFLAG RATES")
total = 50
flag_data = data.get("flag_counts", {})
for variant in ["BASE", "SFT", "DPO"]:
    ch = flag_data.get(f"{variant}:confident_hallucination", 0)
    ts = flag_data.get(f"{variant}:too_shallow", 0)
    print(f"{variant}: confident_hallucination={ch}/50 ({ch/total*100:.0f}%)  too_shallow={ts}/50 ({ts/total*100:.0f}%)")

# Save
output = {
    "per_dimension": {
        v: {
            dim: round(sum(by_variant_dim[v][dim])/len(by_variant_dim[v][dim]), 3)
            for dim in dims
        }
        for v in ["BASE", "SFT", "DPO"]
    }
}
with open("phase11/data/results/score_analysis.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to phase11/data/results/score_analysis.json")
