# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: 20932f6  |  train rows: 81304

## Test metrics
- macro-F1: 0.7732 (95% CI [0.7661, 0.7801])
- recall (hate): 0.7145  |  precision (hate): 0.6556
- ROC-AUC: 0.8704  |  PR-AUC: 0.7405

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.