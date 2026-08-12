# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: 595e63c  |  train rows: 28868

## Test metrics
- macro-F1: 0.7246 (95% CI [0.7116, 0.7353])
- recall (hate): 0.5755  |  precision (hate): 0.6967
- ROC-AUC: 0.8087  |  PR-AUC: 0.7381

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.