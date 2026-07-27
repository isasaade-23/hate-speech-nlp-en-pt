# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: e89602a  |  train rows: 24028

## Test metrics
- macro-F1: 0.6976 (95% CI [0.6833, 0.7111])
- recall (hate): 0.5475  |  precision (hate): 0.6257
- ROC-AUC: 0.7684  |  PR-AUC: 0.6460

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.