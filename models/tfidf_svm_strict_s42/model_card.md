# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: df71154  |  train rows: 43195

## Test metrics
- macro-F1: 0.6816 (95% CI [0.6659, 0.6983])
- recall (hate): 0.4375  |  precision (hate): 0.4035
- ROC-AUC: 0.8346  |  PR-AUC: 0.3856

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.