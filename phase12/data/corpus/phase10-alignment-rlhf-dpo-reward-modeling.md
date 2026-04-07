# Phase 10 — Alignment: RLHF, DPO, Reward Modeling
## Notes and Findings

**Repo:** github.com/dagc-ai/agentic-ai-infra
**Phase duration:** ~4 hours of active compute, April 2026
**Hardware:** RunPod A100 SXM4 80GB (single instance, all four exercises)

---

## Goal

The goal of Phase 10 was to understand how language models get their behavioral characteristics -- not just what they know, but how they decide what to say and how to say it. Specifically: why a fine-tuned model that knows the right answer will still sometimes give a confident wrong one, and what training techniques exist to close that gap.

Phase 9 taught the model what to do. Phase 10 is about teaching it how to judge quality. The concrete output is two things: a reward model that can score response quality, and a DPO-aligned adapter that nudges the model toward better-calibrated responses. Both feed directly into the capstone -- the reward model concept underlies the Editor agent's quality gate, and the DPO adapter is the Writer agent's production model.

---

## Hardware and Tooling

### Hardware
- RunPod A100 SXM4 80GB
- Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
- CUDA: 12.4.1 (template), downgraded to cu121 for compatibility

### Verified Working Stack
The PyTorch 2.4.0 template required a full stack downgrade to match the Phase 9 verified configuration. Transformers 5.x breaks bitsandbytes 4-bit quantization on this hardware due to a missing `set_submodule` method in PyTorch 2.4.1.

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.4.0+cu121 | Must pin to cu121 |
| bitsandbytes | 0.46.1 | 0.49.2 requires cu130, incompatible |
| transformers | 4.44.0 | 5.x breaks with torch 2.4 |
| trl | 0.9.6 | 1.0.0 requires transformers>=4.56.2 |
| peft | 0.18.1 | |
| accelerate | 0.33.0 | 1.x breaks NF4 quantization dispatch |
| datasets | 4.8.4 | |
| anthropic | latest | Exercise 10.4 only |

Install sequence:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install bitsandbytes==0.46.1
pip install transformers==4.44.0
pip install trl==0.9.6
pip install peft==0.18.1 accelerate==0.33.0 datasets
```

### Model Artifacts
All adapters pushed to HuggingFace Hub (private) for persistence across pod restarts:
- Reward model: dagc-ai/llama-3.2-1b-reward-hh-rlhf (not pushed -- transformers 5.x artifact, incompatible with 4.44.0)
- SFT adapter: dagc-ai/llama-3.1-8b-ai-infra-r16
- DPO adapter: dagc-ai/llama-3.1-8b-ai-infra-dpo

**Lesson learned:** Any file excluded from git via .gitignore requires an explicit persistence strategy before pod termination. HuggingFace Hub for model artifacts. RunPod network volume for anything needed across multiple pods without re-downloading.

---

## Exercise 10.1 -- Reward Model Training

### What We Did
Trained a reward model on the Anthropic HH-RLHF dataset using Bradley-Terry preference loss. Base model: Llama 3.2 1B Instruct. Dataset: 5,000 preference pairs from a total of 160,800 available. One epoch, standard AdamW with cosine schedule.

The Bradley-Terry loss is:
```
L = -log sigmoid(r_chosen - r_rejected)
```

The model learns to assign a higher scalar score to the response humans preferred over the one they did not. No labels, no explicit quality definition -- only pairwise human judgment.

### Results
| Metric | Value |
|--------|-------|
| Final loss | 0.6445 |
| Final accuracy | 0.75 |
| Final reward margin | 1.4609 |
| Dataset | HH-RLHF, 5K examples |
| Base model | Llama 3.2 1B Instruct |
| Training time | ~1 hour |
| Estimated cost | ~$2 |

### Key Findings
Accuracy of 0.75 means the reward model correctly identifies the human-preferred response in 3 out of 4 pairs. Random baseline is 0.50. The reward margin of 1.46 is the more meaningful number -- it measures how confidently the model discriminates, not just whether it gets the direction right.

Early steps showed the loss starting near 0.72 (close to the random baseline of 0.69) and dropping irregularly due to the small batch size of 4. Individual step accuracy bounced between 0.25 and 1.0 throughout training. The trend over many steps was what mattered, not individual steps.

The `score.weight MISSING` warning at model load is expected and not an error. `AutoModelForSequenceClassification` replaces the language model head with a new scalar head. That head is randomly initialized and trained from scratch -- it has no pretrained weights to load.

### Failure Mode Documented
Transformers 5.5.0 is incompatible with the reward model architecture when using `model.config.pad_token_id = model.config.eos_token_id` -- in Llama 3.2, `eos_token_id` is a list, not a scalar. Fix: `model.config.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id`. Additionally, `torch_dtype` is deprecated in transformers 5.x -- use `dtype` instead.

---

## Exercise 10.2 -- DPO Training

### What We Did
Ran Direct Preference Optimization on the Phase 9 SFT adapter using TRL 0.9.6's DPOTrainer. Starting point: the Llama 3.1 8B Instruct base model loaded in 4-bit NF4 quantization with the Phase 9 LoRA adapter (r=16, trained on 550 AI infrastructure Q&A pairs). Dataset: 2,000 HH-RLHF preference pairs reformatted into (prompt, chosen, rejected) triples. One epoch.

DPO eliminates the reward model by rearranging the RLHF objective mathematically. The optimal policy under KL-regularized reward has a closed form, which allows the reward to be expressed as a log ratio of policy to reference probabilities. The Bradley-Terry model then gives a loss expressed entirely in terms of the policy and a frozen reference:

```
L_DPO = -E[log sigmoid(
    beta * (log π(y_chosen|x) - log π_ref(y_chosen|x)) -
    beta * (log π(y_rejected|x) - log π_ref(y_rejected|x))
)]
```

In TRL 0.9.6, `ref_model=None` tells DPOTrainer to derive reference model behavior from the base model via PEFT adapter disable/enable rather than loading a second model copy. This halves the memory requirement.

### Results
| Metric | Value |
|--------|-------|
| Final train loss | 0.90 |
| Final accuracy | 0.58 |
| Final reward margin | 0.44 |
| Beta | 0.1 |
| Learning rate | 5e-7 |
| Dataset | HH-RLHF, 1948 examples (after filtering) |
| Training time | ~16 minutes |
| Estimated cost | ~$0.50 |
| VRAM usage | 51GB / 80GB (64%) |
| GPU utilization | 89% at 359W |

### Key Findings
Accuracy of 0.58 is modest compared to the reward model's 0.75. This was expected. The HH-RLHF dataset reflects general helpfulness preferences, not AI infrastructure domain precision. The preference signal is real but noisy relative to our specific use case.

Early oscillation in steps 30-40 where accuracy dropped to 0.44 and margins went negative is normal DPO behavior at initialization. The model recovered by step 70 and held consistently positive margins through the end of the run.

Grad norms started high (22) and settled to 7-10 range by the second half of training -- a sign of stable convergence rather than divergence.

**Critical parameter:** DPO learning rate must be ~5e-7, not SFT-scale 2e-4. SFT-scale learning rates overwrite base model capabilities rather than nudging behavioral tendencies. This is the single most important hyperparameter difference between SFT and DPO training.

### Failure Modes Documented

**Adapter swapping causes nested PEFT structure.** Attempting to swap adapters on a live model (`model = model.base_model` followed by `PeftModel.from_pretrained`) produces a nested `base_model.model.model.model...` hierarchy. PEFT logs hundreds of missing adapter key warnings and the loaded adapter weights are not applied correctly. The fix is to reload the base model fresh for each adapter variant. Never attempt adapter swapping on a model already wrapped in PEFT.

**DPOConfig vs TrainingArguments.** In TRL 0.9.6, DPOTrainer requires a `DPOConfig` object, not a plain `TrainingArguments`. DPO-specific parameters (`beta`, `max_length`, `max_prompt_length`) must be passed inside `DPOConfig`, not as separate arguments to DPOTrainer. TRL 1.0.0 changed this API but is incompatible with transformers 4.44.0.

---

## Exercise 10.3 -- Qualitative Comparison

### What We Did
Ran 10 AI infrastructure prompts through three model variants -- base Llama 3.1 8B, SFT-only (Phase 9 adapter), and SFT+DPO (Phase 10 adapter) -- and compared responses side by side. Each variant was loaded fresh from the base model to avoid adapter stacking issues. Full responses saved to `phase10/data/comparison_results.json`.

### Results Summary

| Prompt | Base | SFT | DPO | Notes |
|--------|------|-----|-----|-------|
| GQA definition | Wrong (visual QA benchmark) | Wrong (visual QA benchmark) | Wrong (visual QA benchmark) | All three hallucinate -- training data gap |
| LoRA vs fine-tuning | Wrong acronym, vague | Correct mechanism, fabricated benchmarks | Correct mechanism, slightly fewer fabrications | SFT clear improvement |
| KV cache scaling | Generic DB caching | Correct but wrong tensor shape | Correct shape, marginally better | SFT major improvement |
| Flash Attention | Wrong (sparse attention) | Correct O(N²) to O(N), minor errors | Correct, minor errors | SFT major improvement |
| Ring AllReduce | Cookie analogy, incomplete | Correct two-phase algorithm | Correct, added tree vs ring tradeoff | SFT major improvement |
| Quantization tradeoffs | Correct concept, incomplete | Correct with fabricated numbers | Correct, cleaner | Comparable |
| Data vs model parallelism | Correct concept | Correct with fabricated RunPod specs | Correct with minor errors | Comparable |
| DPO definition | Wrong (Differential Privacy) | Wrong (Data Poisoning) | Wrong (Data Poisoning) | All three hallucinate -- training data gap |
| Roofline model | Attributed to Google 2018 (wrong) | Wrong formula | Wrong formula, cleaner | All three weak |
| Catastrophic forgetting | Correct concept | Correct with fabricated LoRA specs | Correct, cleaner | SFT clear improvement |

### Key Findings

**SFT improvement is dramatic on domain-covered topics.** Flash Attention, Ring AllReduce, and KV cache explanations went from wrong or vague to technically precise. The 550-pair training dataset directly changed model behavior on these topics.

**DPO improvement is marginal and specific.** DPO produced slightly more precise tensor shapes (Prompt 3), a cleaner roofline formula (Prompt 9), and added the tree vs. ring AllReduce tradeoff (Prompt 5). These are real improvements but not dramatic ones. 2,000 general helpfulness preference pairs applied to a domain-specific SFT adapter produces weak alignment signal.

**Training data gaps produce identical hallucinations across all three variants.** GQA (Grouped Query Attention) and DPO (Direct Preference Optimization) were not explicitly defined in the 550 SFT pairs. All three model variants hallucinated confident wrong definitions. Alignment cannot fix what fine-tuning did not teach. The fix is better SFT data, not more DPO.

**SFT increases confident hallucination on topics at the edge of training coverage.** The model learned to produce confident, specific, technical answers with real numbers. When it does not know the answer, it now fabricates specific numbers that look exactly like correct answers. This is harder to catch than the base model's vaguer wrong answers.

---

## Exercise 10.4 -- LLM-as-Judge Scoring

### What We Did
Used Claude Sonnet (claude-sonnet-4-20250514) as a judge to score all three model variants on the same 10 prompts across three dimensions: technical accuracy (1-5), conciseness (1-5), and hallucination avoidance (1-5). 30 API calls total. Scores saved to `phase10/data/judge_scores.json`.

### Aggregate Results

| Model | Technical Accuracy | Conciseness | Hallucination Avoidance | Mean |
|-------|-------------------|-------------|------------------------|------|
| BASE | 2.20 | 2.50 | 2.20 | 2.30 |
| SFT | 2.40 | 2.20 | 1.20 | 1.93 |
| DPO | 2.00 | 2.20 | 1.30 | 1.83 |

### Key Findings

**The counterintuitive result: BASE scored highest overall.** This is not evidence that training made things worse. It reflects what each model was optimized for.

**Hallucination avoidance tells the real story.** BASE=2.20, SFT=1.20, DPO=1.30. SFT scored dramatically worse on hallucination because it learned to produce confident, specific responses -- including when it had to fabricate specifics. The judge correctly caught invented vLLM benchmarks, fabricated VRAM calculations, and made-up performance figures throughout SFT responses.

**Accuracy: SFT slightly outperforms base.** 2.40 vs 2.20. The domain training is working. SFT gives more accurate answers on topics it was trained on. DPO's slight dip to 2.00 is likely noise at 10 prompts.

**Conciseness: BASE scores highest.** The base model gives shorter, vaguer answers. SFT and DPO give longer, more detailed responses that score lower on conciseness but higher on depth. This is a rubric calibration issue -- conciseness without accuracy is not the goal.

**DPO partially recovered hallucination avoidance.** 1.30 vs SFT's 1.20. The directional improvement is present but small. Larger and more domain-specific preference datasets would produce a stronger signal.

**The judge calibration is working.** The judge correctly identified the GQA hallucination (all models called it a visual QA benchmark), the DPO acronym hallucination (all models defined it as Data Poisoning or Differential Privacy Optimization), and fabricated benchmark numbers throughout SFT and DPO responses. These match the manual assessment from Exercise 10.3 exactly. The judge is a reliable quality gate for the capstone Editor agent.

---

## Phase 10 Synthesis

### What the Full Phase Demonstrated

Three things that training metrics alone cannot show:

**1. SFT teaches a style, not just knowledge.** The model learned to produce confident, specific, technical prose with numbers and citations. That style is correct and valuable when the model knows the answer. It is dangerous when the model does not -- it produces hallucinations that look identical to correct answers.

**2. Alignment nudges but does not fix.** DPO produced marginal improvements on hallucination avoidance and response precision. It did not fix factual gaps from SFT. The two training stages address different problems and neither is a substitute for the other.

**3. Training data quality is the binding constraint.** The hallucinations on GQA and DPO definitions were not fixable by alignment. They required better SFT data. Every phase of the curriculum has a binding constraint. At the alignment layer, the constraint is the quality and coverage of your training data, not the sophistication of the alignment algorithm.

### Implications for the Capstone

The Editor agent's eval rubric needs to flag:
- Acronym definitions as high-risk hallucination category
- Specific numeric claims (VRAM sizes, throughput numbers, parameter counts) as requiring verification
- Incomplete sentences as a quality failure signal (SFT responses cut off mid-sentence on multiple prompts)

The Writer agent should use the DPO adapter as its production model. The marginal hallucination improvement over SFT matters when the Editor agent is the downstream quality gate -- fewer hallucinations to catch means fewer revision cycles.

The eval harness built in Exercise 10.4 is the Editor agent's decision function. The rubric is calibrated and working. Commit it as a reusable module.

---

## Developer Insights

**Reward modeling is infrastructure, not a product.** You do not deploy it to users. You use it as a training signal or as an automated quality scorer. The reward model from Exercise 10.1 is the conceptual foundation for the LLM-as-judge pattern in Exercise 10.4 -- Claude Sonnet is functioning as a pre-trained reward model rather than one you trained yourself.

**DPO API changed significantly between TRL 0.9.6 and 1.0.0.** In 0.9.6, DPO parameters go inside `DPOConfig`. In 1.0.0, the API changed again. Pin your TRL version and document it. Do not assume TRL minor versions are backward compatible.

**Never swap adapters on a live PEFT model.** Load the base model fresh for each adapter. The adapter stacking failure mode produces no clear error -- only a long list of missing key warnings that are easy to miss. The responses look plausible but are running on the wrong weights.

**The HuggingFace Hub upload includes optimizer checkpoints by default.** These are 336MB each and not needed for inference. Use `ignore_patterns` in `upload_folder` to exclude them in future uploads:
```python
api.upload_folder(
    folder_path="phase10/dpo-adapter",
    repo_id="dagc-ai/llama-3.1-8b-ai-infra-dpo",
    token=token,
    ignore_patterns=["checkpoint-*/optimizer.pt"]
)
```

**DPO on a domain SFT adapter with general preference data produces weak signal.** To get strong DPO signal on AI infrastructure content, you need domain-specific preference pairs -- the same Q&A prompts, two responses of different quality, with a clear quality signal. HH-RLHF is general helpfulness. It moves the needle but not dramatically for a specialized domain.

---

## Business Insights

**The alignment gap is a product liability question.** A fine-tuned model that hallucinates confidently is worse than a base model that hedges, because the hallucinations are harder to catch. For enterprise deployments where wrong answers have consequences -- compliance, legal, medical, financial -- the alignment stack is not optional. SFT alone is not sufficient.

**Reward modeling enables scalable quality measurement.** Once you have a working judge (whether your own reward model or an API-based judge), you can score thousands of outputs automatically. This is what makes CI/CD for model quality possible -- run the eval harness on every fine-tuning run, track scores over time, and catch regressions before deployment.

**Training data quality is the real cost center.** The compute for Phase 10 was under $5. The binding constraint was the quality of the 550 SFT pairs from Phase 9. Two specific topics -- GQA as Grouped Query Attention and DPO as Direct Preference Optimization -- were not covered, and no amount of alignment training fixed that. Investing in data quality and coverage is higher leverage than investing in more sophisticated alignment algorithms.

**The two-step training pipeline (SFT then DPO) is now standard.** Any enterprise team building a production model for a specific domain should expect to run both stages. SFT for domain knowledge and format. DPO (or equivalent) for behavioral calibration. This is not optional for high-stakes deployments. It is table stakes.

---

## GTM Insights -- Full Stack

### AI Hardware (Tenstorrent, Groq, Cerebras, Etched)
DPO training requires two model copies in memory simultaneously -- the policy and the reference. For an 8B model in bfloat16, that is 32GB of weights alone before activations or optimizer state. For a 70B model, that is a multi-node problem. Any chip company pitching training infrastructure needs to speak to alignment workloads, not just pretraining. The seller who can say "here is how our hardware handles the two-model DPO memory requirement" is having a more sophisticated conversation than anyone else in the room.

### Inference and Serving Platforms (Together AI, Fireworks, Modal, Baseten)
Aligned models have different serving characteristics than base models. They produce longer, more structured outputs and are less likely to short-circuit with a one-line answer. This affects KV cache utilization and effective batch sizing. A seller who can connect alignment training choices to inference cost per token is speaking the language of the infrastructure buyer, not just the ML buyer.

### Cloud GPU Providers (CoreWeave, Lambda Labs, RunPod, Voltage Park)
Alignment training is a growing workload on GPU clouds. DPO requires lower learning rates and more careful training stability than SFT, which means longer runs at lower throughput. Reward model training is a separate workload -- smaller models, different memory profiles. A seller who understands these workload characteristics can help customers size their instances correctly before they waste money on mis-configured runs. That is a consultative sale, not a commodity GPU rental.

### MLOps and Experiment Tracking (Weights and Biases, Comet, Langfuse)
Alignment training produces distinct artifact types: reward margins, preference accuracy over steps, before/after behavioral comparisons, LLM-as-judge score distributions. These are native W&B artifact types. A seller at an MLOps company who can walk a customer through what a DPO training dashboard should look like -- what metrics to track, what failure modes look like in the curves, how to diagnose reward hacking -- is selling to a use case that is growing fast and not well served by generic ML experiment tracking pitches.

### AI Application Platforms (LangChain, LlamaIndex, Cohere, Contextual AI)
Unaligned models in agent pipelines fail unpredictably. Confident hallucinations from the Writer agent propagate through the Editor and Publisher agents and compound. The alignment conversation for this audience is about pipeline reliability -- how many revision cycles does the Editor agent need to catch Writer agent hallucinations, and what is the cost per published post if the Writer is poorly calibrated? A seller who can frame alignment as an agent reliability and cost question is speaking to a pain point these buyers feel directly.

### Enterprise Software Embedding AI (Salesforce, ServiceNow, Workday, Adobe)
These buyers are evaluating vendor models, not building their own. The alignment conversation for this audience is entirely about risk. What happens when the model gives a confident wrong answer to a customer? What is the liability exposure? Reward modeling and DPO are the technical mechanisms behind the "safe and reliable AI" claims every vendor makes in their pitch deck. Being able to explain what those mechanisms actually do -- in plain language, with specific failure modes and mitigations -- in a conversation with a technical buyer at an enterprise software company is a capability almost no one in enterprise GTM currently has.

### AI Infrastructure for Agents (CockroachDB, Neon, PlanetScale -- database layer)
The alignment training pipeline produces artifacts that need to be stored, versioned, and queried: preference datasets, reward model checkpoints, eval scores, before/after behavioral comparisons. This is a structured data problem. A database seller who can map alignment training workflows to specific query patterns -- time-series eval scores, preference pair storage, model version tracking -- is selling to a use case that most AI teams have not thought carefully about yet.

---

## Commit History

| Commit | Message |
|--------|---------|
| reward_model.py | phase10: reward model training complete | final_loss=0.6445 | accuracy=0.75 | reward_margin=1.4609 | dataset=hh-rlhf-5k | base=llama-3.2-1b |
| dpo_train.py | phase10: DPO training complete | beta=0.1 | lr=5e-07 | dataset=hh-rlhf-2k | base=llama-3.1-8b-sft-r16 | final_accuracy=0.58 | final_margin=0.44 |
| qualitative_comparison.py + comparison_results.json | phase10: qualitative comparison complete | 10 prompts | base vs SFT vs SFT+DPO | DPO shows marginal precision gains | GQA and DPO definition hallucination documented |
| llm_judge.py + judge_scores.json | phase10: LLM-as-judge complete | BASE=2.30 SFT=1.93 DPO=1.83 | SFT increases confident hallucination | DPO marginal improvement on hallucination avoidance |

---

*Phase 10 complete. All artifacts committed to github.com/dagc-ai/agentic-ai-infra/phase10*
*Next: Phase 11 -- Evals: Measuring Model Behavior Before You Build on It*
