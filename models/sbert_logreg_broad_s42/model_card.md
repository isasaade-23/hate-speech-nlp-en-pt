# Model card — sbert_logreg_broad_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: broad
- Seed: 42  |  git: df71154  |  train rows: 43724

## Test metrics
- macro-F1: 0.6928 (95% CI [0.6821, 0.7036])
- recall (hate): 0.5531  |  precision (hate): 0.5827
- ROC-AUC: 0.7650  |  PR-AUC: 0.6031

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.