# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: 595e63c  |  train rows: 28868

## Test metrics
- macro-F1: 0.7288 (95% CI [0.7174, 0.7402])
- recall (hate): 0.6428  |  precision (hate): 0.6586
- ROC-AUC: 0.7978  |  PR-AUC: 0.7162

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.