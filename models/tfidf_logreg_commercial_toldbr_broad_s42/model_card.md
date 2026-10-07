# Model card — tfidf_logreg_commercial_toldbr_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: 0238f5b  |  train rows: 58444

## Test metrics
- macro-F1: 0.7217 (95% CI [0.7151, 0.7288])
- recall (hate): 0.6346  |  precision (hate): 0.7192
- ROC-AUC: 0.7982  |  PR-AUC: 0.7742

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.