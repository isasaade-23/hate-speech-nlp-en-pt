# Model card — tfidf_logreg_hurtlex_strict_s42

- Family: classical  |  Config: tfidf_logreg_hurtlex  |  Policy: strict
- Seed: 42  |  git: 20932f6  |  train rows: 81304

## Test metrics
- macro-F1: 0.7859 (95% CI [0.7793, 0.7928])
- recall (hate): 0.7249  |  precision (hate): 0.6774
- ROC-AUC: 0.8838  |  PR-AUC: 0.7584

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.