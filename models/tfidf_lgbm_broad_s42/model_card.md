# Model card — tfidf_lgbm_broad_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: broad
- Seed: 42  |  git: e89602a  |  train rows: 24028

## Test metrics
- macro-F1: 0.6983 (95% CI [0.6847, 0.7116])
- recall (hate): 0.5513  |  precision (hate): 0.6246
- ROC-AUC: 0.7679  |  PR-AUC: 0.6502

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.