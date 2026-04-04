## Data Preparation

Fine-tuning quality is bounded by data quality, not by training 
hyperparameters. Issues encountered building the Phase 9 dataset:

- JSON encoding failures from unescaped quotes in code examples
- Off-topic pairs from setup/tooling discussions in technical threads
- Meta-references that break pair self-containment
- Minimum length enforcement to filter shallow Q&A

Real-world ratio: data preparation to training is often 3:1 or higher.
The training run is the easy part.