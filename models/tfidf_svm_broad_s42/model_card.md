# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: e89602a  |  train rows: 24028

## Test metrics
- macro-F1: 0.6734 (95% CI [0.6595, 0.6875])
- recall (hate): 0.5250  |  precision (hate): 0.5846
- ROC-AUC: 0.7363  |  PR-AUC: 0.6056

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.