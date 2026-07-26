# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: 5fc983f  |  train rows: 24028

## Test metrics
- macro-F1: 0.7077 (95% CI [0.694, 0.7219])
- recall (hate): 0.5243  |  precision (hate): 0.6715
- ROC-AUC: 0.7891  |  PR-AUC: 0.6821

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.