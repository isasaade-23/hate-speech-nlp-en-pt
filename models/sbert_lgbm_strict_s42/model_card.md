# Model card — sbert_lgbm_strict_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: strict
- Seed: 42  |  git: e89602a  |  train rows: 23499

## Test metrics
- macro-F1: 0.6834 (95% CI [0.6643, 0.7028])
- recall (hate): 0.5197  |  precision (hate): 0.4067
- ROC-AUC: 0.8084  |  PR-AUC: 0.4290

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.