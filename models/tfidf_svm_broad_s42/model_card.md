# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: cac1065  |  train rows: 38772

## Test metrics
- macro-F1: 0.7269 (95% CI [0.7164, 0.7381])
- recall (hate): 0.5581  |  precision (hate): 0.6720
- ROC-AUC: 0.8203  |  PR-AUC: 0.7022

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.