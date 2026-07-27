# Results

All numbers are on the frozen test split, seed 42, reported under two label policies —
**strict** (only explicit hate is positive) and **broad** (offensive folded into hate).

## Model comparison

![Leaderboard](img/leaderboard.png){ width="100%" }

*Test macro-F1 by model and policy. Transformers (coral) top both policies.*

| Policy | Best transformer | Best classical | Δ | McNemar + Holm |
|--------|------------------|----------------|----|----------------|
| strict | XLM-R — **0.750** | tfidf_logreg — 0.709 | +0.041 | p = 0.003 |
| broad  | BERTweet — **0.748** | tfidf_lgbm — 0.698 | +0.050 | p < 0.001 |

The transformer advantage is statistically significant, not an artifact of one split.
BERTimbau (PT) reaches recall-on-hate 0.796 in strict on correctly decoded Portuguese.

## Cross-lingual and cross-domain transfer

![Transfer](img/transfer.png){ width="100%" }

| Experiment (broad) | TF-IDF (word) | SBERT (multilingual) |
|--------------------|:-------------:|:--------------------:|
| EN to PT (zero-shot) | 0.418 | **0.626** |
| PT to EN (zero-shot) | 0.445 | **0.674** |
| tweets to memes (cross-domain) | 0.504 | 0.540 |
| memes to tweets (cross-domain) | 0.501 | 0.550 |

TF-IDF word features share no vocabulary across languages, so cross-lingual recall on hate
falls to near zero; multilingual embeddings place EN and PT in one space and transfer.

## Calibration

The served score should be interpretable as a probability. Best-calibrated models are
`sbert_lgbm` (ECE 0.032 strict, 0.056 broad) and the conservative `tfidf_svm`; the
transformers win on macro-F1 but not on calibration — a trade-off carried into the
[product decision](api.md).

## Identity-term bias

An unintended-bias probe measures the false-positive rate on non-hate text that merely
mentions an identity group (neutral terms, bilingual). Over-flagging is real across all
models — sexual-orientation mentions reach a false-positive rate of 0.75 vs. 0.17
background for TF-IDF — but the transformers show the smallest gaps (~0.26).

## Error analysis

For the best model, **implicit hate** (hateful text with no profanity token) is the largest
false-negative bucket. XLM-R reduces it (183 vs. 206 for the best classical), consistent
with a contextual model catching subtler hate; it over-flags slur-bearing non-hate slightly
more. Portuguese is over-flagged relative to English (a higher false-positive rate).

!!! note "Multi-seed confidence intervals"
    Results shown are single-seed (42). Re-running the transformers over seeds 42/43/44
    and calling `hsc report` produces `leaderboard_agg.csv` with mean ± std per model.
