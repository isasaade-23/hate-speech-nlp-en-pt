# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: 595e63c  |  train rows: 28340

## Test metrics
- macro-F1: 0.7114 (95% CI [0.693, 0.7292])
- recall (hate): 0.4956  |  precision (hate): 0.4899
- ROC-AUC: 0.8380  |  PR-AUC: 0.4967

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.