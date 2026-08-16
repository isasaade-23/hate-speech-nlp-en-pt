# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: c182003  |  train rows: 81304

## Test metrics
- macro-F1: 0.7878 (95% CI [0.7815, 0.7944])
- recall (hate): 0.7073  |  precision (hate): 0.6928
- ROC-AUC: 0.8854  |  PR-AUC: 0.7605

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.