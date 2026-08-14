# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: df71154  |  train rows: 43195

## Test metrics
- macro-F1: 0.7061 (95% CI [0.6896, 0.7221])
- recall (hate): 0.5285  |  precision (hate): 0.4210
- ROC-AUC: 0.8622  |  PR-AUC: 0.4201

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.