# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: 5fc983f  |  train rows: 24028

## Test metrics
- macro-F1: 0.6818 (95% CI [0.6684, 0.6961])
- recall (hate): 0.4931  |  precision (hate): 0.6290
- ROC-AUC: 0.7576  |  PR-AUC: 0.6441

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.