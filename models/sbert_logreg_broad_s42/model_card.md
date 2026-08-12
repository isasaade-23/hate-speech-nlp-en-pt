# Model card — sbert_logreg_broad_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: broad
- Seed: 42  |  git: 595e63c  |  train rows: 28868

## Test metrics
- macro-F1: 0.7192 (95% CI [0.7075, 0.7315])
- recall (hate): 0.6519  |  precision (hate): 0.6351
- ROC-AUC: 0.7896  |  PR-AUC: 0.6817

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.