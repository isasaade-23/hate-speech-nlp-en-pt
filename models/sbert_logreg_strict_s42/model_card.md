# Model card — sbert_logreg_strict_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: strict
- Seed: 42  |  git: df71154  |  train rows: 43195

## Test metrics
- macro-F1: 0.6365 (95% CI [0.6209, 0.6526])
- recall (hate): 0.3981  |  precision (hate): 0.3036
- ROC-AUC: 0.7817  |  PR-AUC: 0.2664

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.