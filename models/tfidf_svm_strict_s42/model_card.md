# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: c182003  |  train rows: 81304

## Test metrics
- macro-F1: 0.7681 (95% CI [0.7609, 0.7754])
- recall (hate): 0.6791  |  precision (hate): 0.6653
- ROC-AUC: 0.8574  |  PR-AUC: 0.7049

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.