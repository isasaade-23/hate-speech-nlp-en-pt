# Model card — sbert_logreg_strict_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: strict
- Seed: 42  |  git: c0ae91a  |  train rows: 23499

## Test metrics
- macro-F1: 0.6690 (95% CI [0.6513, 0.6867])
- recall (hate): 0.5557  |  precision (hate): 0.3661
- ROC-AUC: 0.8077  |  PR-AUC: 0.3851

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.