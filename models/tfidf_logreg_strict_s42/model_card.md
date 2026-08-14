# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: cac1065  |  train rows: 38241

## Test metrics
- macro-F1: 0.7022 (95% CI [0.6846, 0.7176])
- recall (hate): 0.4809  |  precision (hate): 0.4437
- ROC-AUC: 0.8728  |  PR-AUC: 0.4615

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.