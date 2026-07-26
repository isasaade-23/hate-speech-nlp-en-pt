# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: 5fc983f  |  train rows: 23499

## Test metrics
- macro-F1: 0.6986 (95% CI [0.6801, 0.7186])
- recall (hate): 0.4683  |  precision (hate): 0.4748
- ROC-AUC: 0.8425  |  PR-AUC: 0.5002

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.