# Model card — sbert_lgbm_strict_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: strict
- Seed: 42  |  git: cac1065  |  train rows: 38241

## Test metrics
- macro-F1: 0.6441 (95% CI [0.6292, 0.6588])
- recall (hate): 0.4539  |  precision (hate): 0.3107
- ROC-AUC: 0.7868  |  PR-AUC: 0.3250

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.