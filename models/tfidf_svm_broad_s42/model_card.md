# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: f854e84  |  train rows: 81832

## Test metrics
- macro-F1: 0.6512 (95% CI [0.6441, 0.6585])
- recall (hate): 0.6491  |  precision (hate): 0.5984
- ROC-AUC: 0.6969  |  PR-AUC: 0.6206

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.