"""
Phase 11 -- Exercise 3: Generate manual calibration set
Samples 20 responses for manual scoring -- variant hidden to avoid bias.
Output: phase11/data/results/calibration_display.txt
        phase11/data/results/calibration_set.json
"""

import json
import random

random.seed(42)

with open('phase11/data/results/raw_responses.json') as f:
    data = json.load(f)

results = data['results']

# Sample 20 -- roughly balanced across variants
base = [r for r in results if r['variant'] == 'BASE']
sft  = [r for r in results if r['variant'] == 'SFT']
dpo  = [r for r in results if r['variant'] == 'DPO']

sampled = (
    random.sample(base, 7) +
    random.sample(sft, 7) +
    random.sample(dpo, 6)
)
random.shuffle(sampled)

# Print for manual review -- variant hidden to avoid bias
print('MANUAL CALIBRATION SET -- 20 responses')
print('Score each on: technical_accuracy, calibration, mechanistic_depth, audience_calibration (1-5)')
print('=' * 80)

for i, r in enumerate(sampled):
    print(f'\n[{i+1:02d}/20] Category: {r["category"]} | ID: {r["prompt_id"]}')
    print(f'QUESTION: {r["prompt"]}')
    print(f'RESPONSE:\n{r["response"]}')
    print('-' * 80)

# Save sampled IDs with variant for correlation script later
calibration_set = [
    {'prompt_id': r['prompt_id'], 'variant': r['variant']}
    for r in sampled
]
with open('phase11/data/results/calibration_set.json', 'w') as f:
    json.dump(calibration_set, f, indent=2)

print('\nCalibration set saved to phase11/data/results/calibration_set.json')
print('Read each response above and score on all four dimensions before proceeding.')
