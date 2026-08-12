# Model card — sbert_logreg_strict_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: strict
- Seed: 42  |  git: 595e63c  |  train rows: 28340

## Test metrics
- macro-F1: 0.6618 (95% CI [0.6446, 0.678])
- recall (hate): 0.5279  |  precision (hate): 0.3554
- ROC-AUC: 0.8011  |  PR-AUC: 0.3567

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.