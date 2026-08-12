# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: 595e63c  |  train rows: 28340

## Test metrics
- macro-F1: 0.7293 (95% CI [0.7112, 0.747])
- recall (hate): 0.5044  |  precision (hate): 0.5400
- ROC-AUC: 0.8531  |  PR-AUC: 0.5234

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.