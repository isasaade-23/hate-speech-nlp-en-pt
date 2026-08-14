# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: cac1065  |  train rows: 38772

## Test metrics
- macro-F1: 0.7577 (95% CI [0.7471, 0.7678])
- recall (hate): 0.7024  |  precision (hate): 0.6463
- ROC-AUC: 0.8446  |  PR-AUC: 0.7290

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.