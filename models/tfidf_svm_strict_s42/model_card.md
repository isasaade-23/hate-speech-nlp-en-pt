# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: e89602a  |  train rows: 23499

## Test metrics
- macro-F1: 0.6716 (95% CI [0.6521, 0.6909])
- recall (hate): 0.4974  |  precision (hate): 0.3882
- ROC-AUC: 0.7722  |  PR-AUC: 0.3774

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.