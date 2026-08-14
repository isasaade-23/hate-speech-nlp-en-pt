# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: cac1065  |  train rows: 38772

## Test metrics
- macro-F1: 0.7463 (95% CI [0.7352, 0.7562])
- recall (hate): 0.6701  |  precision (hate): 0.6400
- ROC-AUC: 0.8338  |  PR-AUC: 0.7154

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.