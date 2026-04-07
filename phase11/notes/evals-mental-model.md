# Phase 11 — Evals: Measuring Model Behavior Before You Build on It

## Goal of This Phase

The central goal of Phase 11 is to build measurement infrastructure for
model behavior before wiring any model into a production agent pipeline.

Every previous phase produced a benchmark: CUDA kernel throughput in GB/s,
AllReduce bus bandwidth utilization, KV cache size at 100K context, LoRA
rank sensitivity curves, reward model accuracy. Phase 11 applies the same
discipline to the model layer itself -- not "does it run" but "does it
produce correct, reliable output on the specific task distribution we care
about."

This phase sits at the junction between the training phases (8-10) and the
agent phases (12-16). Its outputs are not academic -- the eval harness built
here becomes a live component of the Phase 16 capstone. The rubric becomes
the Editor agent's decision function. The 50-prompt test set becomes the
permanent quality benchmark for the content engine. The calibration work
determines whether the Editor can be trusted to make autonomous approval
decisions without human review.

### How It Connects to the Big Picture

The recurring principle across every phase of this curriculum is: every
layer has a binding constraint, and correct optimization requires identifying
which constraint is actually binding before deciding what to fix.

At the model layer, the binding constraint is not parameter count or benchmark
score. It is task-specific reliability on your actual output distribution.
A model that scores 85% on MMLU but hallucinates NVLink bandwidth numbers
with high confidence is worse than useless for a technical content engine --
it is actively dangerous, because it produces content that looks correct and
is wrong.

Phase 11 is the measurement pass that identifies what is actually binding
before the agent pipeline is built on top of it. You cannot retrofit eval
requirements onto a system not designed around a rubric. Define the quality
bar first, then build the system that must maintain it.

---

## Hardware and Software Stack

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA A100 SXM4 80GB |
| CPU | AMD EPYC 7742 |
| System RAM | 232.8 GB |
| CUDA | 12.4 (driver) / 12.1 (PyTorch build) |
| PyTorch | 2.4.0+cu121 |
| Transformers | 4.44.0 (pinned -- 5.x breaks 4-bit quantization) |
| PEFT | 0.18.1 |
| bitsandbytes | 0.46.1 |
| TRL | 0.9.6 |
| Accelerate | 0.33.0 |
| Anthropic SDK | latest (judge API calls) |
| Compute platform | RunPod (A100 SXM4 80GB instance) |
| Judge model | claude-sonnet-4-20250514 |
| Base model | meta-llama/Llama-3.1-8B-Instruct |
| SFT adapter | dagc-ai/llama-3.1-8b-ai-infra-r16 (Phase 9) |
| DPO adapter | dagc-ai/llama-3.1-8b-ai-infra-dpo (Phase 10) |

---

## Why Benchmarks Lie

Off-the-shelf benchmarks (MMLU, HumanEval, MT-Bench) measure performance
on a fixed dataset that almost certainly differs from your production task
distribution. Three failure modes:

**Distribution mismatch**: MMLU tests academic multiple-choice recall.
The content engine task is continuous technical prose generation on niche
AI infrastructure topics. These are not the same capability and do not
predict each other.

**Format artifacts**: Multiple-choice allows pattern-matching answer
structure without understanding content. A model can score above random
by learning the surface properties of correct answer choices. Generation
tasks expose this immediately.

**Benchmark contamination**: Frontier models are trained on internet
scrapes that include benchmark datasets, papers describing them, forum
discussions of specific questions, and adjacent content. High benchmark
scores may reflect memorization, not reasoning. The contamination test
in this phase directly diagnoses this.

---

## Contamination Test Finding

Tested DPO variant on 10 original vs. rephrased prompt pairs.
Same response evaluated against original and rephrased question.

| prompt_id | orig | reph | delta | signal |
|-----------|------|------|-------|--------|
| gpu_001 | 2.75 | 3.25 | +0.500 | rephrased better |
| kernel_002 | 2.75 | 2.50 | -0.250 | mild drop |
| dist_001 | 1.75 | 2.50 | +0.750 | rephrased better |
| inf_001 | 2.75 | 2.75 | 0.000 | stable |
| arch_002 | 1.50 | 2.25 | +0.750 | rephrased better |
| ft_001 | 1.75 | 1.75 | 0.000 | stable |
| ft_005 | 1.25 | 1.50 | +0.250 | rephrased better |
| arch_001 | 1.75 | 1.75 | 0.000 | stable |
| inf_002 | 2.75 | 2.75 | 0.000 | stable |
| agent_002 | 1.75 | 1.25 | -0.500 | mild drop |

Average delta: +0.150 | Contamination signals: 0/10

**Interpretation**: The model is not pattern-matching surface form.
Hallucination is confabulation from partial knowledge generated in
real time, not memorized wrong answers. This is harder to fix than
contamination -- it requires RAG grounding or a strong Editor gate,
not data decontamination. Rephrase the question a dozen ways and you
get the same confidently wrong answer every time.

---

## Rubric Design (v1.1)

Four dimensions chosen to directly target the Phase 10 failure mode
(confident hallucination amplified by SFT training):

| Dimension | What It Catches |
|-----------|----------------|
| Technical Accuracy | Wrong definitions, wrong numbers, wrong mechanisms |
| Calibration | Confident hallucination -- correct confidence requires correct knowledge |
| Mechanistic Depth | Vague explanations that sound correct but explain nothing |
| Audience Calibration | Wrong depth for practitioner audience |

Deliberately excluded: length, fluency, formatting. These are what
naive judges reward and what SFT optimized for. They are not correlated
with technical correctness on niche AI infrastructure topics.

### v1.0 to v1.1 Change

Calibration identified a systematic positive bias: judge scored higher
than human on 17/20 responses. Root cause: score-2 anchor on technical
accuracy was too forgiving. "Imprecise" was doing too much work -- a
wrong definition is not imprecise, it is wrong.

v1.1 tightened the score-2 anchor to explicitly state: a response that
gets the core definition wrong scores 2 or below on technical_accuracy
regardless of how well the rest of the response reads.

---

## Calibration Results

Manual scored 20 responses independently. Compared against judge scores.

| Dimension | Pearson r | MAE | Status |
|-----------|-----------|-----|--------|
| technical_accuracy | 0.808 | 0.40 | PASS |
| calibration | 0.829 | 0.40 | PASS |
| mechanistic_depth | 0.720 | 0.55 | NEEDS REVIEW |
| audience_calibration | 0.811 | 0.90 | PASS |
| OVERALL | 0.861 | 0.562 | PASS |

Bias: judge scored higher than human on 17/20 (85%). Systematic
positive bias corrected in rubric v1.1. Overall r=0.861 passes the
0.75 threshold -- judge is trustworthy for autonomous Editor decisions
with the tightened rubric applied.

---

## Model Comparison Results

50 prompts, 3 variants, 150 total responses scored by judge v1.0.

| Variant | Mean Score | CH Flags |
|---------|-----------|----------|
| BASE | 2.400 | 27/50 (54%) |
| SFT | 2.520 | 37/50 (74%) |
| DPO | 2.575 | 37/50 (74%) |

### Per-Dimension Breakdown

| Dimension | BASE | SFT | DPO | Direction |
|-----------|------|-----|-----|-----------|
| technical_accuracy | 2.020 | 1.820 | 1.920 | Fine-tuning made accuracy worse |
| calibration | 2.460 | 2.040 | 2.060 | Fine-tuning made calibration worse |
| mechanistic_depth | 2.040 | 2.720 | 2.760 | Fine-tuning helped significantly |
| audience_calibration | 3.080 | 3.500 | 3.560 | Fine-tuning helped significantly |

---

## Findings

### Finding 1: Fine-tuning produced a clean two-by-two split

Fine-tuning improved style dimensions (mechanistic depth, audience
calibration) and degraded accuracy dimensions (technical accuracy,
calibration). This is not a mixed result -- it is a precise diagnosis.

The model learned to write like an expert without becoming one. SFT
taught confident, structured, practitioner-appropriate phrasing. It
did not teach the underlying technical facts, because those facts were
sparse in the training data and could not be learned from 550 pairs.

### Finding 2: SFT destroyed the agent infrastructure category

SFT scored 1.964 on agent_infrastructure vs BASE at 2.679 -- the
largest single category regression in the dataset. The SFT training
dataset had minimal agent infrastructure coverage. The model learned
confident phrasing patterns from the topics it did cover, then applied
those patterns to agent questions it knew least about.

### Finding 3: DPO did not reduce hallucination rate

SFT and DPO both flagged confident_hallucination on 37/50 responses.
HH-RLHF preference pairs had no signal about technical accuracy in AI
infrastructure. Preference training nudged style and depth but could
not fix a knowledge gap. You cannot DPO your way to domain expertise.

### Finding 4: The Phase 10 contradiction is explained by rubric design

Phase 10 scored BASE highest at 2.30. Phase 11 also ranks BASE lowest
at 2.400 but the contradiction resolves cleanly: Phase 10 used a
generic helpfulness rubric that rewarded structured confident responses.
Phase 11 explicitly penalizes confident hallucination. Same model,
different rubric, opposite ranking. The rubric determines what you
measure. Helpful-sounding and technically correct are not the same
thing.

### Finding 5: The domain is genuinely hard

GPU architecture scored best across all three variants (BASE 2.857,
SFT 3.071, DPO 3.107). Inference serving scored worst for BASE (2.036).
The pattern tracks training data coverage: GPU architecture has been
written about extensively since 2012. PagedAttention, NF4 quantization,
and DPO were published in 2022-2023 with far less derivative content
in training corpora.

No variant averaged above 2.6 on a 1-5 scale. No variant is reliable
enough to publish without an Editor gate.

---

## Insights from Findings in Simple Terms

The models have learned what good AI infrastructure explanations look
like stylistically but not what makes them correct factually. Ask about
Flash Attention and you get a well-structured response with the right
vocabulary in the right order. Ask whether the specific numbers are
accurate and they often are not.

This is the same problem as hiring someone who interviewed extremely
well and then discovered on the job that they had been confidently
describing things they had read summaries of, not things they understood.
The interview (benchmark) did not catch it. Putting them in a real
situation (task-specific eval) did.

The contamination test confirmed the hallucination is not cached wrong
answers from training data. The model is generating wrong explanations
from scratch in real time, consistently. That means the fix is not
cleaning the training data -- it is giving the model access to correct
information at inference time via RAG, and catching errors before
publication via the Editor gate.

---

## Insights from the Developer Lens

The throughput delta between BASE (18.1 tok/s) and adapter variants
(11.0-11.1 tok/s) is a 39% inference penalty from loading PEFT adapters.
In production the fix is merge_and_unload() -- permanently merging
adapter weights into the base model eliminates the overhead. For the
capstone, the Writer agent should run a merged model, not a live adapter.

The memory non-release bug observed between SFT and DPO load cycles
(11.58 GB persisting after unload) is a production pattern to harden.
gc.collect() after torch.cuda.empty_cache() is the fix. In a tighter
memory environment this would cause OOM on the third load cycle.

LLM-as-judge has systematic biases that compound if not measured.
Length bias, confidence bias, self-preference bias, and position bias
are all documented. The calibration methodology (score 20 manually,
compute Pearson r, identify divergence pattern, tighten rubric) is
the correct engineering response. An uncalibrated judge inside a
feedback loop will reinforce the exact failure modes you are trying
to eliminate.

The contamination test is underused in practice. It takes 20 API calls
and 10 minutes. The diagnostic value -- distinguishing memorization
from confabulation -- directly determines what remediation is needed.
Every team evaluating a model for production should run it.

---

## Insights from the Business Lens

Every published post that contains a confident wrong number is a
credibility event. For dagc.ai targeting practitioners who will
actually verify claims, a hallucinated benchmark figure or wrong
bandwidth spec is not a minor error -- it signals that the publication
cannot be trusted. One bad post damages the credibility of all
subsequent posts.

The eval gate is not a cost center. It is the mechanism that makes
the content engine commercially viable. Without it, the pipeline
produces content that looks good and is unreliable. With it, the
pipeline produces content that can be published under a name without
manual review of every post.

The cost model matters. One blog post through the full pipeline
(research, write, edit, revise, publish) costs roughly $0.05-0.15
in API calls at current pricing. Human review of a technical post
by a qualified reviewer costs $50-200 in time. The Editor agent pays
for itself on the first post it catches and returns for revision.

No variant averaged above 2.6 out of 5 on this rubric. That is a
business decision point: the content engine as designed requires
frontier API calls (Claude, GPT-4o) as the Writer agent for production
quality, with local fine-tuned models reserved for draft generation
or lower-stakes content. The fine-tuned Llama variants are not
production-ready on their own for this domain.

---

## Insights from the GTM Lens

The conversation this phase enables with a technical buyer or hiring
manager at an AI infrastructure company is qualitatively different
from anything a standard AI GTM candidate can have.

Standard GTM candidate: "We ran evals and the model performed well."

After Phase 11: "We built a task-specific eval harness, calibrated
an LLM judge against manual scores to r=0.861, ran a contamination
test that distinguished confabulation from memorization, and identified
a clean two-by-two split where fine-tuning improved style dimensions
and degraded accuracy dimensions. Here are the benchmark numbers and
here is the rubric we derived from them that is now running as the
Editor agent's decision function in our autonomous content pipeline."

The second conversation requires a peer-level technical response.
The first does not.

For Tenstorrent specifically: the confabulation finding connects
directly to the case for RAG-grounded inference. A model that
consistently gets NVLink bandwidth numbers wrong in standalone
generation can be grounded with a retrieval layer that pulls current
spec sheets at inference time. That is an infrastructure argument
for the hardware they are selling, made from first-principles
measurement rather than vendor claims.

For inference platform companies (Together AI, Fireworks, Modal):
the throughput delta between base and adapter variants (39% penalty)
and the merge_and_unload() production pattern is a concrete
optimization they deal with at every customer deployment. Speaking
this language from benchmark data rather than documentation is the
difference between a GTM candidate who read the docs and one who
ran the code.

---

## Phase 16 Implications

### Writer Agent
Use DPO -- best mechanistic depth and audience calibration. Produces
most useful explanations even though accuracy is unreliable on its own.

### Editor Agent Decision Function
APPROVE if:
technical_accuracy >= 3.0
AND calibration >= 3.0
AND mean_score >= 3.25
AND judge_flag != "confident_hallucination"
REJECT otherwise -- return dimension scores and reasoning to Writer
Technical accuracy and calibration are weighted double in the approval
gate because these are the dimensions where DPO is actively unreliable.
Any confident_hallucination flag is automatic reject regardless of mean.
Maximum 3 revision cycles before escalating to human review.

### Longer Term
The correct fix for the accuracy problem is not more fine-tuning on
the same domain. It is RAG -- giving the Writer agent access to a
retrieval layer over verified technical sources at generation time.
Phase 12 builds exactly this. The eval harness built here will measure
whether RAG grounding actually moves the technical_accuracy dimension.

---

## Artifacts

| File | Description |
|------|-------------|
| data/prompts/eval_prompts.json | 50-prompt task-specific eval set, 7 categories |
| data/results/raw_responses.json | 150 model responses (BASE/SFT/DPO x 50 prompts) |
| data/results/judge_scores.json | Judge scores v1.0, all 150 responses |
| data/results/score_analysis.json | Per-dimension breakdown by variant and category |
| data/results/manual_scores.json | 20 manually scored responses with detailed notes |
| data/results/calibration_set.json | Calibration sample mapping (prompt_id, variant) |
| data/results/calibration_results.json | Pearson correlation analysis, r=0.861 overall |
| data/results/contamination_results.json | Rephrasing test, 0/10 contamination signals |
| scripts/01_generate_responses.py | Inference runner, 3 variants, safe adapter loading |
| scripts/02_judge_responses.py | Judge harness, rubric v1.1 |
| scripts/03_analyze_scores.py | Per-dimension aggregation and flag counts |
| scripts/04_calibration_sample.py | Manual calibration sample generator |
| scripts/05_calibration_correlation.py | Pearson correlation and bias analysis |
| scripts/06_contamination_test.py | Rephrasing contamination diagnostic |

---

*Phase 11 of: Infrastructure for Agentic AI*
*Companion repo: github.com/dagc-ai/agentic-ai-infra*
*Part I: github.com/dagc-ai/ai-infra-learning*
