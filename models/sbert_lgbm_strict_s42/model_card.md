# Model card — sbert_lgbm_strict_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: strict
- Seed: 42  |  git: 595e63c  |  train rows: 28340

## Test metrics
- macro-F1: 0.6778 (95% CI [0.6591, 0.6961])
- recall (hate): 0.4370  |  precision (hate): 0.4306
- ROC-AUC: 0.8032  |  PR-AUC: 0.3906

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.