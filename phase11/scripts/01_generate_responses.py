"""
Phase 11 -- Exercise 1: Generate responses for all three model variants
Variants: BASE, SFT (r16 adapter), DPO (dpo adapter)
Output: phase11/data/results/raw_responses.json

Usage:
    export HF_TOKEN="your-token"
    python phase11/scripts/01_generate_responses.py
"""

import os
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL_ID   = "meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER_ID  = "dagc-ai/llama-3.1-8b-ai-infra-r16"
DPO_ADAPTER_ID  = "dagc-ai/llama-3.1-8b-ai-infra-dpo"

PROMPTS_PATH    = "phase11/data/prompts/eval_prompts.json"
OUTPUT_PATH     = "phase11/data/results/raw_responses.json"

MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.1   # low but not zero -- avoids greedy artifacts
TOP_P           = 0.9

# ── Credentials ───────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN='your-token'")

# ── Quantization config (same as Phase 9/10) ─────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_base_model():
    """Load quantized base model and tokenizer."""
    print(f"\nLoading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )
    model.eval()
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


def load_adapter(base_model, adapter_id):
    """
    Load a PEFT adapter on top of the base model.
    Returns a NEW PeftModel -- does not mutate base_model in place.
    Safe pattern: avoids the nested model.model.model hierarchy
    from Phase 10 failure mode.
    """
    print(f"  Loading adapter: {adapter_id}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_id,
        token=HF_TOKEN
    )
    peft_model.eval()
    return peft_model


def unload_model(model):
    """Fully release GPU memory before loading next variant."""
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  GPU memory after unload: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def format_prompt(tokenizer, prompt_text):
    """Apply Llama 3 Instruct chat template consistently across all variants."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical expert in AI infrastructure. "
                "Explain concepts clearly and accurately for a practitioner audience. "
                "Be specific -- use concrete numbers and mechanisms, not vague generalities."
            )
        },
        {
            "role": "user",
            "content": prompt_text
        }
    ]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


def generate_response(model, tokenizer, prompt_text):
    """Generate a single response. Returns text and timing."""
    formatted = format_prompt(tokenizer, prompt_text)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    # Decode only the generated tokens, not the prompt
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    tokens_generated = len(generated_ids)
    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0

    return response_text, elapsed, tokens_generated, tokens_per_sec


def run_variant(model, tokenizer, prompts, variant_name):
    """Run all 50 prompts through a single model variant."""
    print(f"\nRunning variant: {variant_name} ({len(prompts)} prompts)")
    results = []

    for i, p in enumerate(prompts):
        print(f"  [{i+1:02d}/{len(prompts)}] {p['id']}", end="", flush=True)
        response, elapsed, n_tokens, tok_per_sec = generate_response(
            model, tokenizer, p["prompt"]
        )
        print(f" -- {n_tokens} tokens, {tok_per_sec:.1f} tok/s")

        results.append({
            "prompt_id":       p["id"],
            "category":        p["category"],
            "prompt":          p["prompt"],
            "variant":         variant_name,
            "response":        response,
            "tokens_generated": n_tokens,
            "elapsed_sec":     round(elapsed, 3),
            "tokens_per_sec":  round(tok_per_sec, 1),
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load prompts
    with open(PROMPTS_PATH) as f:
        prompt_data = json.load(f)
    prompts = prompt_data["prompts"]
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")

    all_results = []

    # ── Variant 1: BASE ───────────────────────────────────────────────────────
    model, tokenizer = load_base_model()
    base_results = run_variant(model, tokenizer, prompts, "BASE")
    all_results.extend(base_results)
    unload_model(model)

    # ── Variant 2: SFT ───────────────────────────────────────────────────────
    # Reload base fresh -- avoids any state contamination from BASE run
    model, tokenizer = load_base_model()
    sft_model = load_adapter(model, SFT_ADAPTER_ID)
    sft_results = run_variant(sft_model, tokenizer, prompts, "SFT")
    all_results.extend(sft_results)
    unload_model(sft_model)
    unload_model(model)

    # ── Variant 3: DPO ───────────────────────────────────────────────────────
    model, tokenizer = load_base_model()
    dpo_model = load_adapter(model, DPO_ADAPTER_ID)
    dpo_results = run_variant(dpo_model, tokenizer, prompts, "DPO")
    all_results.extend(dpo_results)
    unload_model(dpo_model)
    unload_model(model)

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "phase": "11",
        "description": "Raw model responses -- BASE, SFT, DPO -- 50 prompts each",
        "total_responses": len(all_results),
        "generation_config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature":    TEMPERATURE,
            "top_p":          TOP_P,
        },
        "results": all_results
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(all_results)} responses to {OUTPUT_PATH}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    from collections import defaultdict
    by_variant = defaultdict(list)
    for r in all_results:
        by_variant[r["variant"]].append(r["tokens_per_sec"])

    print("\nThroughput summary:")
    for variant, tps_list in sorted(by_variant.items()):
        avg = sum(tps_list) / len(tps_list)
        print(f"  {variant}: avg {avg:.1f} tok/s across {len(tps_list)} prompts")


if __name__ == "__main__":
    main()
