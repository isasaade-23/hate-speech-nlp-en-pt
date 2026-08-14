# Model card — tfidf_logreg_broad_s42

- Family: classical  |  Config: tfidf_logreg  |  Policy: broad
- Seed: 42  |  git: df71154  |  train rows: 43724

## Test metrics
- macro-F1: 0.7311 (95% CI [0.7208, 0.7411])
- recall (hate): 0.6178  |  precision (hate): 0.6302
- ROC-AUC: 0.8175  |  PR-AUC: 0.6962

## Intended use & limitations
Research classifier for EN/PT social-media hate speech. Probabilistic; not a
moderation oracle. Training-data licenses restrict commercial use (see
methodology/data_provenance.md). See methodology/limitations.md.