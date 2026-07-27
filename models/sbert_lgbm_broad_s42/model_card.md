# Model card — sbert_lgbm_broad_s42

- Family: classical  |  Config: sbert_lgbm  |  Policy: broad
- Seed: 42  |  git: c0ae91a  |  train rows: 24028

## Test metrics
- macro-F1: 0.6950 (95% CI [0.6812, 0.7088])
- recall (hate): 0.6723  |  precision (hate): 0.5633
- ROC-AUC: 0.7734  |  PR-AUC: 0.6553

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.