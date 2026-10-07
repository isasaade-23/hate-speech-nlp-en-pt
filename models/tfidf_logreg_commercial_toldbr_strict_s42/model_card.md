# Model card — tfidf_logreg_commercial_toldbr_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: 0238f5b  |  train rows: 57915

## Test metrics
- macro-F1: 0.7699 (95% CI [0.7629, 0.7771])
- recall (hate): 0.6517  |  precision (hate): 0.6887
- ROC-AUC: 0.8479  |  PR-AUC: 0.7353

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.