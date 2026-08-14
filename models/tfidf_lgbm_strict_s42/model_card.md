# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: df71154  |  train rows: 43195

## Test metrics
- macro-F1: 0.6950 (95% CI [0.6798, 0.7119])
- recall (hate): 0.4973  |  precision (hate): 0.4071
- ROC-AUC: 0.8443  |  PR-AUC: 0.4172

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.