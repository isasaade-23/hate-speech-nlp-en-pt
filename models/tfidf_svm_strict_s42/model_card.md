# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: 595e63c  |  train rows: 28340

## Test metrics
- macro-F1: 0.6967 (95% CI [0.6791, 0.7147])
- recall (hate): 0.4736  |  precision (hate): 0.4614
- ROC-AUC: 0.8271  |  PR-AUC: 0.4597

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.