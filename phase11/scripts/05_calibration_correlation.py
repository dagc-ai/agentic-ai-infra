"""
Phase 11 -- Exercise 3: Correlation between manual scores and LLM judge scores
Computes Pearson correlation per dimension and overall.
Output: printed report + phase11/data/results/calibration_results.json

Usage:
    python phase11/scripts/05_calibration_correlation.py
"""

import json
import math
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────────────────────

with open('phase11/data/results/manual_scores.json') as f:
    manual_data = json.load(f)

with open('phase11/data/results/judge_scores.json') as f:
    judge_data = json.load(f)

with open('phase11/data/results/calibration_set.json') as f:
    calibration_set = json.load(f)

# ── Build lookup: (prompt_id, variant) -> judge scores ───────────────────────

judge_lookup = {}
for r in judge_data['results']:
    if r.get('judge_scores'):
        key = (r['prompt_id'], r['variant'])
        judge_lookup[key] = r['judge_scores']

# Map calibration set order to variants
# calibration_set is ordered to match manual_scores response_num
cal_map = {
    i+1: {'prompt_id': c['prompt_id'], 'variant': c['variant']}
    for i, c in enumerate(calibration_set)
}

# ── Pair manual vs judge scores ───────────────────────────────────────────────

dims = ['technical_accuracy', 'calibration', 'mechanistic_depth', 'audience_calibration']

paired = []
unmatched = []

for s in manual_data['scores']:
    num     = s['response_num']
    cal_entry = cal_map.get(num)
    if not cal_entry:
        unmatched.append(num)
        continue

    prompt_id = cal_entry['prompt_id']
    variant   = cal_entry['variant']
    key       = (prompt_id, variant)

    if key not in judge_lookup:
        unmatched.append(num)
        continue

    judge_scores = judge_lookup[key]

    paired.append({
        'response_num': num,
        'prompt_id':    prompt_id,
        'variant':      variant,
        'manual':       {d: s[d] for d in dims},
        'judge':        {d: judge_scores[d] for d in dims},
        'manual_mean':  round(sum(s[d] for d in dims) / 4, 3),
        'judge_mean':   round(sum(judge_scores[d] for d in dims) / 4, 3),
        'notes':        s.get('notes', '')
    })

print(f"Paired: {len(paired)}/20 | Unmatched: {len(unmatched)}")
if unmatched:
    print(f"  Unmatched response nums: {unmatched}")

# ── Pearson correlation ───────────────────────────────────────────────────────

def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den = math.sqrt(sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y))
    return round(num/den, 3) if den > 0 else None

def mean_abs_error(x, y):
    return round(sum(abs(xi-yi) for xi,yi in zip(x,y)) / len(x), 3)

# Per-dimension correlation
print("\n" + "="*60)
print("CALIBRATION RESULTS: Manual vs Judge")
print("="*60)
print(f"\n{'Dimension':<28} {'Pearson r':>10} {'MAE':>8} {'Status':>12}")
print("-"*60)

dim_results = {}
all_manual_means = [p['manual_mean'] for p in paired]
all_judge_means  = [p['judge_mean']  for p in paired]

for dim in dims:
    manual_scores = [p['manual'][dim] for p in paired]
    judge_scores  = [p['judge'][dim]  for p in paired]

    r   = pearson(manual_scores, judge_scores)
    mae = mean_abs_error(manual_scores, judge_scores)
    status = "PASS (r>=0.75)" if r and r >= 0.75 else "NEEDS REVIEW"

    print(f"{dim:<28} {r:>10} {mae:>8} {status:>12}")
    dim_results[dim] = {'pearson_r': r, 'mae': mae, 'status': status}

# Overall mean correlation
r_overall = pearson(all_manual_means, all_judge_means)
mae_overall = mean_abs_error(all_manual_means, all_judge_means)
status_overall = "PASS (r>=0.75)" if r_overall and r_overall >= 0.75 else "NEEDS REVIEW"

print("-"*60)
print(f"{'OVERALL (mean score)':<28} {r_overall:>10} {mae_overall:>8} {status_overall:>12}")

# ── Per-response comparison ───────────────────────────────────────────────────

print("\n" + "="*60)
print("PER-RESPONSE COMPARISON")
print("="*60)
print(f"{'#':>3} {'prompt_id':<14} {'variant':<6} {'manual':>8} {'judge':>8} {'delta':>8}")
print("-"*60)

large_divergence = []
for p in paired:
    delta = round(p['manual_mean'] - p['judge_mean'], 2)
    flag  = " <--" if abs(delta) >= 1.0 else ""
    print(f"{p['response_num']:>3} {p['prompt_id']:<14} {p['variant']:<6} {p['manual_mean']:>8.2f} {p['judge_mean']:>8.2f} {delta:>8.2f}{flag}")
    if abs(delta) >= 1.0:
        large_divergence.append(p)

# ── Divergence analysis ───────────────────────────────────────────────────────

if large_divergence:
    print(f"\nLARGE DIVERGENCES (delta >= 1.0): {len(large_divergence)} responses")
    for p in large_divergence:
        delta = round(p['manual_mean'] - p['judge_mean'], 2)
        direction = "human higher" if delta > 0 else "judge higher"
        print(f"\n  [{p['response_num']:02d}] {p['prompt_id']} ({p['variant']}) -- delta={delta} ({direction})")
        print(f"  Notes: {p['notes'][:120]}...")

# ── Bias diagnosis ────────────────────────────────────────────────────────────

judge_higher = sum(1 for p in paired if p['judge_mean'] > p['manual_mean'])
human_higher = sum(1 for p in paired if p['manual_mean'] > p['judge_mean'])
equal        = sum(1 for p in paired if p['manual_mean'] == p['judge_mean'])

print(f"\nBIAS ANALYSIS:")
print(f"  Judge scored higher than human: {judge_higher}/20 ({judge_higher/20*100:.0f}%)")
print(f"  Human scored higher than judge: {human_higher}/20 ({human_higher/20*100:.0f}%)")
print(f"  Equal:                          {equal}/20")

if judge_higher > 14:
    print("  >> Judge has systematic positive bias -- consider stricter rubric anchors")
elif human_higher > 14:
    print("  >> Human is systematically harsher -- consider whether rubric is too strict")
else:
    print("  >> No strong systematic directional bias detected")

# ── Save results ──────────────────────────────────────────────────────────────

output = {
    'phase': '11',
    'n_paired': len(paired),
    'n_unmatched': len(unmatched),
    'dimension_results': dim_results,
    'overall': {
        'pearson_r': r_overall,
        'mae': mae_overall,
        'status': status_overall
    },
    'bias': {
        'judge_higher': judge_higher,
        'human_higher': human_higher,
        'equal': equal
    },
    'large_divergences': [
        {
            'response_num': p['response_num'],
            'prompt_id': p['prompt_id'],
            'variant': p['variant'],
            'manual_mean': p['manual_mean'],
            'judge_mean': p['judge_mean'],
            'delta': round(p['manual_mean'] - p['judge_mean'], 2),
            'notes': p['notes']
        }
        for p in large_divergence
    ],
    'paired_scores': paired
}

with open('phase11/data/results/calibration_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to phase11/data/results/calibration_results.json")
