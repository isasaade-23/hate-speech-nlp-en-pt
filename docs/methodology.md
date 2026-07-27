# Methodology

Every experiment is config-driven and seeded; each run records its config, git SHA, and
seed. The full decision trail lives in the repository's `methodology/` folder: a
chronological decision log, data provenance with licenses, the label-mapping rationale, the
experiment log, and known limitations.

## Data

Four Kaggle datasets are harmonized to one binary schema.

| Source | Language | Domain | Rows | Original label | License |
|--------|----------|--------|------|----------------|---------|
| memotion | EN | meme (OCR) | ~6,992 | offensive (4 levels) | citation-only |
| tweets_ip | EN | tweet | ~21,009 | 1/2/3 (inferred) | uncertain |
| pt_fortuna | PT | web comment | ~5,670 | binary 0/1 | uncertain |
| multioff | EN | meme (OCR) | ~743 | offensive / not | Apache-2.0 |

Language and domain are confounded (PT = comments; EN = tweets + memes), so every claim is
broken down by source, and a dedicated transfer experiment separates language from domain.

## Label policies (the scientific crux)

The boundary between *offensive* and *hateful* is treated as a variable, not a fixed choice.
Every result is reported under **strict** (only explicit hate is positive) and **broad**
(offensive folded into hate). Reporting both turns label heterogeneity into a sensitivity
analysis instead of a hidden assumption.

## Anti-leakage (non-negotiable)

- Exact deduplication (hash of normalized text) plus near-duplicate clustering
  (MinHash/LSH over character n-grams).
- Split **before any fit**, grouped by duplicate cluster so no paraphrase crosses
  train/test, stratified by (language, source, label), frozen with a content hash.
- Vectorizers, scalers, and class weights are fit on the training split only. A CI test
  asserts zero text overlap across splits.

## The unlabeled tweets source

One English source ships labels 1/2/3 with no legend. A probe (class distribution + a
profanity-lexicon rate per class) supports the mapping 1 = hate, 2 = offensive, 3 = neutral,
and opens a gate that keeps the source out of the primary corpus until the decision is
recorded — a config change, not a code change.

## A data-integrity bug worth documenting

The Portuguese source was being decoded as **latin-1** when the file is **UTF-8**, which
corrupted every accent. A byte-level audit caught it (`C3 A9` is UTF-8 `é`, not `Ã©`), and
the whole pipeline was rebuilt on corrected text — which alone lifted zero-shot EN to PT
transfer by ~4 points. Lesson folded into the provenance protocol: validate encoding by
bytes, never by how a terminal renders a glyph.

## Evaluation

Primary metric is macro-F1; secondary are ROC-AUC, PR-AUC, and recall-on-hate (the
ethically important number). The decision threshold is tuned on validation to maximize
macro-F1 — identically for every model, classical and neural — so the comparison is fair
under class imbalance. Significance is a paired McNemar test with Holm correction, plus
bootstrap confidence intervals. Calibration, identity-term bias, and cross-lingual /
cross-domain transfer complete the picture.
