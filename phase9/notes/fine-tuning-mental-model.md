# Phase 9: Fine-Tuning — SFT, LoRA, QLoRA

## Goal

Take a pretrained base model and shape its behavior for a specific task without 
retraining it from scratch. Understand what fine-tuning actually changes in the 
weights, why full fine-tuning is almost never the right choice, and how LoRA and 
QLoRA make fine-tuning accessible on a single GPU. Build and run a real fine-tuning 
pipeline on domain-specific data generated from the curriculum itself.

---

## Exercise 1: Building the Training Dataset

### Theory and Relevance to Agentic AI Infrastructure

Before a model can be shaped for a specific task, you need training data that 
represents that task distribution. For agentic systems, this means instruction/response 
pairs that reflect the exact behavioral target -- the format, depth, and vocabulary 
the agent needs to produce reliably. Data quality is the binding constraint on 
fine-tuning quality, not the training loop.

### What We Did

Generated 589 raw Q&A pairs from 6 curriculum threads (Phases 1-6), built a 
reproducible pipeline to validate, filter, deduplicate, and shuffle them into a 
550-pair training dataset. Encountered and solved real data pipeline problems: 
JSON encoding failures from unescaped quotes, off-topic pairs from setup discussions, 
meta-references that broke pair self-containment.

### Key Learnings

- Data preparation to training ratio in production is often 3:1 or higher
- JSON encoding failures are a common failure mode when generating synthetic data 
  from LLM conversations -- unescaped quotes inside code examples break parsers
- Off-topic pairs contaminate domain signal -- SSH key pairs in a GPU architecture 
  dataset will teach the model the wrong things
- Minimum response length enforcement (75 words) filters shallow Q&A that adds 
  noise without signal
- A reproducible build script (build_dataset.py) is as important as the data itself 
  -- anyone should be able to regenerate the dataset from source files

### Developer Perspective

The pipeline you build for 550 pairs is the same pipeline production teams run at 
100,000 pairs -- ingestion, generation, quality filtering, deduplication, versioning. 
The difference is orchestration (Airflow, Prefect) and scale (distributed inference 
for generation, MinHash LSH for deduplication). The concepts are identical.

### Business Perspective

Enterprise fine-tuning projects fail more often on data quality than on model 
selection or hyperparameters. A customer with 10 years of support tickets, internal 
documentation, or expert analyst reports has latent training signal that competitors 
cannot replicate. The data pipeline that unlocks that signal is the actual 
competitive moat -- not the model.

---

## Exercise 2: QLoRA Fine-Tuning End to End

### Theory and Relevance to Agentic AI Infrastructure

Full fine-tuning a 70B model requires 840GB+ of memory -- weights, gradients, and 
optimizer state. This is only accessible to organizations running multi-node clusters. 
LoRA makes fine-tuning accessible by decomposing weight updates as delta_W = B @ A, 
where B is (d x r) and A is (r x d) with rank r << d. Only B and A receive gradient 
updates -- the base model is frozen. QLoRA extends this by quantizing the frozen base 
weights to NF4 (4-bit), reducing a 16GB model to 4GB in memory while keeping adapter 
training in BF16 for numerical stability.

The mathematical assumption: fine-tuning changes are low-rank. The space of behavioral 
changes useful for a specific task is low-dimensional -- the model already knows 
language, you are teaching it what to do with that knowledge. That behavioral target 
lives in a much smaller subspace than the full weight space.

### What We Did

Fine-tuned Llama 3.1 8B Instruct on 550 AI infrastructure Q&A pairs using QLoRA 
at rank=16. Training ran on a single A100 80GB for 5 minutes 10 seconds, 3 epochs, 
102 steps. Final training loss: 1.533 (down from 2.51 at step 1). Adapter saved at 
161MB -- 0.52% of total model parameters.

### Key Learnings

- `get_peft_model()` freezes the base weights and injects LoRA wrapper modules -- 
  the base model stays in memory and participates in every forward pass, gradients 
  only flow through A and B
- bitsandbytes is the most dependency-sensitive library in this stack -- CUDA version, 
  PyTorch version, and bitsandbytes version must all align. Mismatches produce 
  cryptic runtime errors, not clear version warnings
- Transformers 5.x introduced breaking API changes (set_submodule, SFTConfig) -- 
  pin your stack before running: torch==2.4.0+cu121, transformers==4.44.0, 
  trl==0.9.6, bitsandbytes==0.46.1
- The base model consumes 19GB VRAM in NF4. The adapter and optimizer state add 
  ~2GB. 80GB A100 is at 23% utilization -- plenty of headroom for larger datasets
- Never hardcode credentials in scripts that commit to public repos. Use 
  os.environ.get("HF_TOKEN") and export the token in the shell before running

### Developer Perspective

One LoRA adapter is one behavioral target. To serve multiple behaviors from one base 
model, load one copy of the base model in VRAM and hot-swap adapters at request time. 
This is the production pattern at companies running fine-tuned models at scale -- 
one 16GB base model, many 50-200MB adapters, each specializing behavior for a 
different task or customer.

### Business Perspective

QLoRA makes fine-tuning accessible without a cluster. A single A100 80GB ($1.49/hr 
on RunPod) can fine-tune a 70B model. The training run for this exercise cost under 
$0.15. The capability that costs OpenAI and Anthropic millions to develop at scale 
is now accessible to any engineering team with a credit card and a well-curated 
dataset. The moat is the data, not the compute.

---

## Exercise 3: Rank Sensitivity Experiment

### Theory and Relevance to Agentic AI Infrastructure

Rank r determines the dimensionality of the weight update subspace. Higher rank means 
more expressive adapters -- more capacity to capture complex behavioral changes. The 
question is where diminishing returns begin: at what rank does adding parameters stop 
improving quality? This matters for production agentic systems because adapter size 
affects storage, transfer time, and hot-swap latency when serving multiple adapters 
against one base model.

### What We Did

Trained 5 adapters at r=4, 8, 16, 32, 64 on identical data with identical 
hyperparameters. Recorded trainable parameters, final loss, and training time for each.

| Rank | Trainable Params | Final Loss | Training Time |
|------|-----------------|------------|---------------|
| r=4  | 10,485,760      | 1.8055     | 312s          |
| r=8  | 20,971,520      | 1.6771     | 308s          |
| r=16 | 41,943,040      | 1.5330     | 310s          |
| r=32 | 83,886,080      | 1.3784     | 309s          |
| r=64 | 167,772,160     | 1.1982     | 312s          |

### Key Learnings

- Loss improves consistently with rank -- no plateau observed on this dataset
- Training time is rank-invariant -- adapter parameters are negligible relative 
  to the frozen base model, so doubling rank costs nothing in wall clock time
- The expected crossover at r=16 did not materialize -- the behavioral target 
  (technical AI infrastructure content with specific numbers and reasoning patterns) 
  has higher intrinsic dimensionality than simple instruction-following tasks
- Practical sweet spot for this dataset: r=32 -- best loss-to-parameter tradeoff 
  before adapter size doubles again with marginal return
- Parameters scale exactly linearly with rank: r=4 to r=8 doubles params, r=8 to 
  r=16 doubles again. The math holds precisely

### Developer Perspective

Rank is a hyperparameter you tune against your specific task and dataset. Common 
default of r=16 is reasonable for simple behavioral tasks (format compliance, persona) 
but undershoots for technically dense domains. Run the experiment before committing 
to a rank -- the cost is negligible since training time doesn't change.

### Business Perspective

Adapter size matters at deployment scale. A 50MB adapter hot-swaps in milliseconds. 
A 320MB adapter introduces latency when switching between customer-specific 
configurations. For enterprise deployments with hundreds of customer-specific adapters, 
the rank decision has real infrastructure implications -- storage, transfer bandwidth, 
and swap latency all scale with adapter size.

---

## Exercise 4: Qualitative Before/After Comparison

### Theory and Relevance to Agentic AI Infrastructure

Loss curves tell you the model is learning. They do not tell you whether the model 
is learning the right things. Qualitative evaluation closes this gap -- running 
identical prompts against the base model and the fine-tuned model and documenting 
the behavioral difference. For agentic systems, behavioral reliability on the target 
task distribution is what matters, not benchmark scores.

### What We Did

Ran 10 AI infrastructure prompts against base Llama 3.1 8B (no adapter) and against 
the r=32 fine-tuned adapter. Documented responses side by side in 
before-after-comparison.md.

### Key Learnings

- Fine-tuning measurably improved precision on well-covered topics: arithmetic 
  intensity, Flash Attention, Ring AllReduce, tiled matmul, roofline model -- 
  all produced quantitative, accurate responses matching training data vocabulary
- The base model hallucinated the roofline model as a psychology framework by 
  Daniel Kahneman. The fine-tuned model correctly described it as a GPU performance 
  analysis tool. Fine-tuning corrected a concrete factual hallucination
- The base model described Tenstorrent as a Chinese chip designer -- wrong. The 
  fine-tuned model gave a conceptually accurate answer about the architectural bet
- Fine-tuning introduced a new failure mode: the fine-tuned model sometimes generates 
  follow-up questions instead of answers on certain prompts. This is a training data 
  artifact -- some pairs in the dataset used a Socratic format that the model learned 
  to reproduce
- Topics with thin training data coverage (Chinchilla scaling laws) remained weak 
  in the fine-tuned model -- fine-tuning does not conjure knowledge that wasn't in 
  the training data
- Fine-tuning does not reliably inject new facts -- it shapes behavior and response 
  patterns. For factual grounding, RAG is the right tool (Phase 12)

### Developer Perspective

Always run a qualitative comparison before declaring a fine-tune successful. Loss 
going down does not mean behavior improved -- the model can overfit to format patterns 
without improving factual accuracy, or it can learn failure modes from inconsistent 
training data. The qualitative comparison is the actual ground truth.

### Business Perspective

For a customer evaluating a fine-tuned model, loss curves are meaningless. What 
matters is: does it answer my domain questions more accurately than the base model? 
Does it maintain the right format consistently? Does it fail gracefully or 
confidently wrong? The qualitative comparison document is the artifact you show 
to a customer -- not the training logs.

---

## GTM Insight

Fine-tuning is a wedge into enterprise AI conversations that most GTM candidates 
cannot use effectively. The typical framing -- "you can customize the model for your 
use case" -- is vague and unpersuasive to a technical buyer. The precise framing is:

Fine-tuning addresses three specific problems: format compliance at scale (the model 
produces the exact output schema your pipeline expects, reliably, without prompt 
engineering overhead), behavioral consistency (the model maintains a specific voice 
or domain vocabulary across thousands of outputs), and cost/latency reduction 
(a fine-tuned 8B model can match a prompted 70B model on specific tasks at a fraction 
of the inference cost).

It does not address factual grounding for new knowledge -- that requires RAG. It 
does not replace alignment -- a fine-tuned model can be made to behave badly just 
as easily as a base model. Knowing the boundaries of what fine-tuning solves and 
what it does not is the signal that distinguishes a technical GTM candidate from 
one who has read the marketing materials.

The data pipeline is the actual competitive moat in enterprise fine-tuning. The 
model is a commodity. The curated, domain-specific, high-quality training dataset 
that took months and domain expertise to build -- that is what competitors cannot 
replicate.
