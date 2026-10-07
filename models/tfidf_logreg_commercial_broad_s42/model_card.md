# Model card — tfidf_logreg_commercial_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: 0238f5b  |  train rows: 43589

## Test metrics
- macro-F1: 0.7040 (95% CI [0.6969, 0.7114])
- recall (hate): 0.5937  |  precision (hate): 0.7114
- ROC-AUC: 0.7564  |  PR-AUC: 0.7512

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.