# Model card — sbert_lgbm_broad_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: broad
- Seed: 42  |  git: cac1065  |  train rows: 38772

## Test metrics
- macro-F1: 0.7117 (95% CI [0.7013, 0.7234])
- recall (hate): 0.6275  |  precision (hate): 0.5911
- ROC-AUC: 0.7933  |  PR-AUC: 0.6434

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.