# Model card — tfidf_logreg_commercial_strict_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: strict
- Seed: 42  |  git: 0238f5b  |  train rows: 43059

## Test metrics
- macro-F1: 0.7686 (95% CI [0.7614, 0.7759])
- recall (hate): 0.6363  |  precision (hate): 0.6975
- ROC-AUC: 0.8416  |  PR-AUC: 0.7341

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.