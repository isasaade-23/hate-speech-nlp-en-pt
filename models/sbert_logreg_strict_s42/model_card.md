# Model card — sbert_logreg_strict_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: strict
- Seed: 42  |  git: cac1065  |  train rows: 38241

## Test metrics
- macro-F1: 0.6234 (95% CI [0.6068, 0.6391])
- recall (hate): 0.3532  |  precision (hate): 0.2971
- ROC-AUC: 0.7806  |  PR-AUC: 0.2669

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.