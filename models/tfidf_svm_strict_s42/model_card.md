# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: 20932f6  |  train rows: 81304

## Test metrics
- macro-F1: 0.7644 (95% CI [0.757, 0.7718])
- recall (hate): 0.6628  |  precision (hate): 0.6673
- ROC-AUC: 0.8548  |  PR-AUC: 0.6992

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.