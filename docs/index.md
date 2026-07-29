# Bilingual Hate-Speech Detection (EN / PT)

![Banner](img/banner.svg){ width="100%" }

Binary hate-speech classification for **English and Portuguese** social-media text, with
language detection at the input. The project pursues two goals at once: a **reproducible
research artifact** (every table and figure is generated from code) and a **deployable
product** (a calibrated inference API with a documented model-selection decision).

## What is compared

A fair, leakage-safe comparison of two model families under one protocol:

- **Classical / tabular.** TF-IDF (word + character n-grams) and frozen multilingual
  sentence embeddings (SBERT), feeding Logistic Regression, Linear SVM, and LightGBM.
- **Transformers.** XLM-RoBERTa (bilingual), BERTimbau (PT), and BERTweet (EN).

## Headline findings

- **Transformers win, significantly.** XLM-R reaches macro-F1 0.750 (strict) and BERTweet
  0.748 (broad), ~4 points over the best classical baseline, confirmed by a paired McNemar
  test with Holm correction (p = 0.003 strict; p < 0.001 broad).
- **Cross-lingual transfer.** Trained on English and tested zero-shot on Portuguese,
  TF-IDF collapses while multilingual SBERT transfers (macro-F1 0.42 to 0.63 EN to PT).
- **Leakage-safe by construction.** Exact + near-duplicate (MinHash/LSH) deduplication and
  a group-stratified frozen split, enforced by a CI test.
- **Beyond a single number.** Probability calibration, an identity-term bias probe, and a
  qualitative error analysis (implicit hate is the main blind spot).
- **A data-integrity bug, found and fixed.** The Portuguese source was decoded as latin-1
  when it is UTF-8; a byte-level audit caught it and the pipeline was rebuilt on corrected text.

See [Results](results.md) for the full comparison, [Methodology](methodology.md) for the
decision trail, and [API and product](api.md) for the deployable interface.

!!! warning "Responsible use"
    This is a probabilistic research classifier, not a moderation oracle. It reflects the
    biases of its training data and should support, never replace, human review.
