## Data Preparation

Fine-tuning quality is bounded by data quality, not by training 
hyperparameters. Issues encountered building the Phase 9 dataset:

- JSON encoding failures from unescaped quotes in code examples
- Off-topic pairs from setup/tooling discussions in technical threads
- Meta-references that break pair self-containment
- Minimum length enforcement to filter shallow Q&A

Real-world ratio: data preparation to training is often 3:1 or higher.
The training run is the easy part.

## Dataset: phase9/data/training_dataset.json

| Phase | Raw Pairs | After Filter |
|-------|-----------|--------------|
| Phase 1 - GPU Architecture | 93 | 77 |
| Phase 2 - CUDA Kernels | 90 | 89 |
| Phase 3 - Triton | 99 | 90 |
| Phase 4 - Distributed | 95 | 94 |
| Phase 5 - Inference | 106 | 106 |
| Phase 6 - Hardware | 106 | 104 |
| **Total** | **589** | **550** |

Filtered: 20 off-topic pairs (SSH setup, tooling noise)
Deduplicated: 10 near-duplicate pairs across phases
Mean response length: 128 words
Build script: phase9/scripts/build_dataset.py

## Rank Sensitivity Experiment Results

| Rank | Trainable Params | Final Loss | Training Time |
|------|-----------------|------------|---------------|
| r=4  | 10,485,760      | 1.8055     | 312s          |
| r=8  | 20,971,520      | 1.6771     | 308s          |
| r=16 | 41,943,040      | 1.5330     | 310s          |
| r=32 | 83,886,080      | 1.3784     | 309s          |
| r=64 | 167,772,160     | 1.1982     | 312s          |

Key findings:
- Loss improves consistently with rank -- no clear plateau on this dataset
- Training time is rank-invariant -- adapter params are negligible vs. frozen base
- Practical sweet spot: r=32 for best loss/parameter tradeoff
- r=64 continues improving -- suggests the behavioral target has higher 
  intrinsic dimensionality than typical instruction-following tasks
- Dataset size (550 pairs, narrow technical domain) likely drives the 
  continued improvement at high rank
