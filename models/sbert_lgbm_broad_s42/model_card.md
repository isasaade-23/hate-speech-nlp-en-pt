# Model card — sbert_lgbm_broad_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: broad
- Seed: 42  |  git: e89602a  |  train rows: 24028

## Test metrics
- macro-F1: 0.6898 (95% CI [0.6764, 0.7038])
- recall (hate): 0.5288  |  precision (hate): 0.6202
- ROC-AUC: 0.7698  |  PR-AUC: 0.6360

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.