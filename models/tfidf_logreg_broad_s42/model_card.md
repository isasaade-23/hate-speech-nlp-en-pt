# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: f854e84  |  train rows: 81832

## Test metrics
- macro-F1: 0.7521 (95% CI [0.7455, 0.7592])
- recall (hate): 0.7247  |  precision (hate): 0.7215
- ROC-AUC: 0.8352  |  PR-AUC: 0.8027

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.