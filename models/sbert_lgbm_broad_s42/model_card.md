# Model card — sbert_lgbm_broad_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: broad
- Seed: 42  |  git: 595e63c  |  train rows: 28868

## Test metrics
- macro-F1: 0.7210 (95% CI [0.7094, 0.7333])
- recall (hate): 0.6702  |  precision (hate): 0.6299
- ROC-AUC: 0.8044  |  PR-AUC: 0.7102

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.