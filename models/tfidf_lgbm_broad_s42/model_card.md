# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: f854e84  |  train rows: 81832

## Test metrics
- macro-F1: 0.7438 (95% CI [0.737, 0.7511])
- recall (hate): 0.7104  |  precision (hate): 0.7149
- ROC-AUC: 0.8210  |  PR-AUC: 0.7922

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.