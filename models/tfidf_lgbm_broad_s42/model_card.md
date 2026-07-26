# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: 5fc983f  |  train rows: 24028

## Test metrics
- macro-F1: 0.7108 (95% CI [0.6968, 0.7253])
- recall (hate): 0.5437  |  precision (hate): 0.6624
- ROC-AUC: 0.7817  |  PR-AUC: 0.6807

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.