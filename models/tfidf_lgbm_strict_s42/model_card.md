# Model card — tfidf_lgbm_strict_s42

- Family: classical  |  Config: tfidf_lgbm  |  Policy: strict
- Seed: 42  |  git: e89602a  |  train rows: 23499

## Test metrics
- macro-F1: 0.7065 (95% CI [0.6881, 0.7252])
- recall (hate): 0.5575  |  precision (hate): 0.4452
- ROC-AUC: 0.8200  |  PR-AUC: 0.4567

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.