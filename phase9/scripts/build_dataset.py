import json
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "../data/raw")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../data/training_dataset.json")

REJECT_KEYWORDS = [
    "ssh", "vast.ai", "authorized_keys", "ssh-ed25519",
    "pip install", "git clone", "api key", "conda",
    "as we discussed", "in this thread", "we saw that"
]

FILES = [
    "phase1_gpu_architecture.json",
    "phase2_cuda_kernels.json",
    "phase3_triton.json",
    "phase4_distributed.json",
    "phase5_inference.json",
    "phase6_hardware.json",
]

def is_on_topic(pair):
    text = (pair["instruction"] + pair["response"]).lower()
    return not any(kw in text for kw in REJECT_KEYWORDS)

def is_long_enough(pair, min_words=75):
    return len(pair["response"].split()) >= min_words

def load_and_filter(filepath):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    before = len(data)
    data = [d for d in data if is_on_topic(d) and is_long_enough(d)]
    after = len(data)
    if before != after:
        print(f"  Filtered {before - after} pairs")
    return data

def deduplicate(pairs, similarity_threshold=0.8):
    seen_instructions = []
    unique = []
    for pair in pairs:
        words = set(pair["instruction"].lower().split())
        is_dup = False
        for seen in seen_instructions:
            overlap = len(words & seen) / max(len(words | seen), 1)
            if overlap >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(pair)
            seen_instructions.append(words)
    return unique

all_pairs = []

for filename in FILES:
    filepath = os.path.join(RAW_DIR, filename)
    print(f"Loading {filename}...")
    pairs = load_and_filter(filepath)
    print(f"  {len(pairs)} pairs after filtering")
    all_pairs.extend(pairs)

print(f"\nTotal before dedup: {len(all_pairs)}")
all_pairs = deduplicate(all_pairs)
print(f"Total after dedup: {len(all_pairs)}")

random.seed(42)
random.shuffle(all_pairs)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_pairs, f, indent=2, ensure_ascii=False)

print(f"\nWrote {len(all_pairs)} pairs to {OUTPUT_FILE}")

# Final stats
lengths = [len(d["response"].split()) for d in all_pairs]
print(f"Min response length: {min(lengths)} words")
print(f"Max response length: {max(lengths)} words")
print(f"Mean response length: {sum(lengths)//len(lengths)} words")