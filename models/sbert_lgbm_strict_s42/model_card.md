# Model card — sbert_lgbm_strict_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: strict
- Seed: 42  |  git: df71154  |  train rows: 43195

## Test metrics
- macro-F1: 0.6487 (95% CI [0.6337, 0.6641])
- recall (hate): 0.4185  |  precision (hate): 0.3249
- ROC-AUC: 0.7897  |  PR-AUC: 0.3126

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.