# Model card — tfidf_logreg_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: 5fc983f  |  train rows: 23499

## Test metrics
- macro-F1: 0.7169 (95% CI [0.6992, 0.7352])
- recall (hate): 0.5832  |  precision (hate): 0.4588
- ROC-AUC: 0.8569  |  PR-AUC: 0.5246

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.