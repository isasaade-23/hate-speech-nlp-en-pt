# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: 595e63c  |  train rows: 28868

## Test metrics
- macro-F1: 0.7459 (95% CI [0.7343, 0.7571])
- recall (hate): 0.6558  |  precision (hate): 0.6865
- ROC-AUC: 0.8199  |  PR-AUC: 0.7473

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.