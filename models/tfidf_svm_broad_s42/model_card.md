# Model card — tfidf_svm_broad_s42

- Family: classical  |  Config: tfidf_svm  |  Policy: broad
- Seed: 42  |  git: df71154  |  train rows: 43724

## Test metrics
- macro-F1: 0.7169 (95% CI [0.7067, 0.7268])
- recall (hate): 0.5870  |  precision (hate): 0.6169
- ROC-AUC: 0.7933  |  PR-AUC: 0.6630

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.