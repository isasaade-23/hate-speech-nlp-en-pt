# Model card — sbert_lgbm_strict_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: strict
- Seed: 42  |  git: c0ae91a  |  train rows: 23499

## Test metrics
- macro-F1: 0.6886 (95% CI [0.6702, 0.7075])
- recall (hate): 0.5026  |  precision (hate): 0.4265
- ROC-AUC: 0.8141  |  PR-AUC: 0.4149

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.