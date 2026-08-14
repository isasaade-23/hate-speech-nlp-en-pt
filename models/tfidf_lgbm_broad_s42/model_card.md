# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: df71154  |  train rows: 43724

## Test metrics
- macro-F1: 0.7294 (95% CI [0.719, 0.7394])
- recall (hate): 0.5836  |  precision (hate): 0.6496
- ROC-AUC: 0.8123  |  PR-AUC: 0.6899

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.