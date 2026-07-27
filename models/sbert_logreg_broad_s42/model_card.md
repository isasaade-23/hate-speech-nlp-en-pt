# Model card — sbert_logreg_broad_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: broad
- Seed: 42  |  git: 450ee3c  |  train rows: 24028

## Test metrics
- macro-F1: 0.6854 (95% CI [0.6711, 0.6997])
- recall (hate): 0.5675  |  precision (hate): 0.5866
- ROC-AUC: 0.7652  |  PR-AUC: 0.6325

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.