# Model card — sbert_logreg_broad_s42

- Family: classical  |  Config: sbert_logreg  |  Policy: broad
- Seed: 42  |  git: cac1065  |  train rows: 38772

## Test metrics
- macro-F1: 0.6943 (95% CI [0.6832, 0.7058])
- recall (hate): 0.5808  |  precision (hate): 0.5787
- ROC-AUC: 0.7799  |  PR-AUC: 0.6086

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.