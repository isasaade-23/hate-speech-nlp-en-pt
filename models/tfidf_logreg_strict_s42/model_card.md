# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: e89602a  |  train rows: 23499

## Test metrics
- macro-F1: 0.7094 (95% CI [0.6892, 0.7298])
- recall (hate): 0.4631  |  precision (hate): 0.5143
- ROC-AUC: 0.8413  |  PR-AUC: 0.4870

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.