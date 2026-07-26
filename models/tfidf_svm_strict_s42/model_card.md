# Model card — tfidf_svm_strict_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: strict
- Seed: 42  |  git: 5fc983f  |  train rows: 23499

## Test metrics
- macro-F1: 0.6891 (95% CI [0.672, 0.7067])
- recall (hate): 0.5043  |  precision (hate): 0.4267
- ROC-AUC: 0.8075  |  PR-AUC: 0.4269

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.