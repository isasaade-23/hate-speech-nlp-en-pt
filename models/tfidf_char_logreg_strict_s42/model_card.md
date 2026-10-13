# Model card — tfidf_char_logreg_strict_s42

- Family: classical  |  Config: tfidf_char_logreg  |  Policy: strict
- Seed: 42  |  git: 998f5a5  |  train rows: 81304

## Test metrics
- macro-F1: 0.7749 (95% CI [0.7681, 0.7821])
- recall (hate): 0.6971  |  precision (hate): 0.6695
- ROC-AUC: 0.8733  |  PR-AUC: 0.7323

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.