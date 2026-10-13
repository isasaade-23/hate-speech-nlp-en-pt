# Model card — tfidf_char_logreg_broad_s42

- Family: classical  |  Config: tfidf_char_logreg  |  Policy: broad
- Seed: 42  |  git: 998f5a5  |  train rows: 81832

## Test metrics
- macro-F1: 0.7408 (95% CI [0.7339, 0.7476])
- recall (hate): 0.7045  |  precision (hate): 0.7129
- ROC-AUC: 0.8205  |  PR-AUC: 0.7816

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.