# Model card — sbert_logreg_strict_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: strict
- Seed: 42  |  git: e89602a  |  train rows: 23499

## Test metrics
- macro-F1: 0.6645 (95% CI [0.6468, 0.6827])
- recall (hate): 0.4871  |  precision (hate): 0.3762
- ROC-AUC: 0.8005  |  PR-AUC: 0.3923

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.