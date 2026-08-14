# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: cac1065  |  train rows: 38241

## Test metrics
- macro-F1: 0.7065 (95% CI [0.6886, 0.723])
- recall (hate): 0.4965  |  precision (hate): 0.4464
- ROC-AUC: 0.8604  |  PR-AUC: 0.4629

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.