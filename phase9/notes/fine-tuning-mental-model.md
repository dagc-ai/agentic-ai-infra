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
