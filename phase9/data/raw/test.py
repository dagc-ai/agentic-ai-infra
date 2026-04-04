import json

with open("data/raw/phase6_hardware.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"{len(data)} pairs")
print(f"Min response length: {min(len(d['response'].split()) for d in data)} words")
print(f"Max response length: {max(len(d['response'].split()) for d in data)} words")
print(f"\nFirst pair preview:")
print(f"  Q: {data[0]['instruction'][:80]}")
print(f"  A: {data[0]['response'][:80]}...")