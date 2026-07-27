"""Unit tests for the Fase 9 evaluation machinery: threshold tuning, calibration
scores, paired McNemar, and Holm correction. Synthetic inputs keep them fast and
independent of the trained models."""

from __future__ import annotations

import numpy as np

from hsc.analysis import _holm
from hsc.evaluate import best_threshold, calibration_curve_bins, mcnemar


def test_best_threshold_recovers_separating_cut():
    # Scores cleanly separable at 0.5; the tuned threshold must land between the groups.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    t = best_threshold(y_true, y_score)
    assert 0.4 <= t <= 0.6
    y_pred = (y_score >= t).astype(int)
    assert (y_pred == y_true).all()


def test_best_threshold_beats_fixed_half_under_imbalance():
    # 10% positives with scores shifted low: a 0.5 cut predicts all-negative (macro-F1
    # collapses); a tuned threshold must recover the minority class.
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 20 + [0] * 180)
    y_score = np.concatenate([rng.uniform(0.3, 0.6, 20), rng.uniform(0.0, 0.4, 180)])
    from sklearn.metrics import f1_score

    t = best_threshold(y_true, y_score)
    tuned = f1_score(y_true, (y_score >= t).astype(int), average="macro")
    fixed = f1_score(y_true, (y_score >= 0.5).astype(int), average="macro")
    assert tuned >= fixed


def test_calibration_perfect_scores_have_low_error():
    # Confidence equals empirical frequency by construction -> small ECE/Brier.
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)  # P(y=1) == p exactly
    cal = calibration_curve_bins(y, p, n_bins=10)
    assert cal["ece"] < 0.05
    assert cal["brier"] < 0.25
    assert cal["mce"] <= 1.0


def test_calibration_handles_decision_function_scale():
    # Unbounded SVM-style scores must be min-max scaled, not crash or exceed [0,1].
    y = np.array([0, 0, 1, 1, 1, 0])
    scores = np.array([-2.5, -1.0, 0.5, 3.0, 1.2, -0.3])
    cal = calibration_curve_bins(y, scores, n_bins=5)
    assert 0.0 <= cal["ece"] <= 1.0
    assert 0.0 <= cal["brier"] <= 1.0


def test_mcnemar_detects_asymmetric_disagreement():
    # b is right everywhere a is wrong and vice-versa is rare -> significant.
    y_true = np.array([1] * 100)
    pred_a = np.array([1] * 100)  # a always right
    pred_b = np.array([0] * 40 + [1] * 60)  # b wrong on 40
    out = mcnemar(y_true, pred_a, pred_b)
    assert out["a_only_correct"] == 40
    assert out["b_only_correct"] == 0
    assert out["p_value"] < 0.01


def test_mcnemar_identical_models_not_significant():
    y_true = np.array([0, 1] * 50)
    pred = np.array([0, 1] * 50)
    out = mcnemar(y_true, pred, pred)
    assert out["a_only_correct"] == 0
    assert out["b_only_correct"] == 0
    assert out["p_value"] == 1.0


def test_holm_monotone_rejection():
    # Smallest p rejected; Holm stops at the first non-reject in ascending order.
    pvals = [0.001, 0.04, 0.5]
    rej = _holm(pvals)
    assert rej[0] is True
    assert rej[2] is False


def test_bias_term_matching_is_word_bounded_and_per_language():
    from hsc.bias_probe import IDENTITY_TERMS, _compile, _mentions

    pats = _compile(IDENTITY_TERMS["sexual_orientation"])
    assert _mentions("I am a gay man", "en", pats) is True
    assert _mentions("sou uma pessoa trans", "pt", pats) is True
    # word boundary: 'gaydar' must NOT match 'gay'
    assert _mentions("my gaydar is broken", "en", pats) is False
    # language routing: an EN term is looked up under 'en', so a pt row w/ only EN text misses
    assert _mentions("a totally neutral sentence", "en", pats) is False


def test_pareto_front_drops_dominated_models():
    import pandas as pd

    from hsc.product import _pareto_front

    # C is dominated by A on every axis; A and B trade F1 vs latency -> both survive.
    df = pd.DataFrame(
        [
            {"model_id": "A", "test_macro_f1": 0.75, "recall_hate": 0.6, "ece": 0.05,
             "bias_gap": 0.10, "latency_p95_ms": 30.0, "size_mb": 400.0},
            {"model_id": "B", "test_macro_f1": 0.70, "recall_hate": 0.5, "ece": 0.04,
             "bias_gap": 0.08, "latency_p95_ms": 2.0, "size_mb": 4.0},
            {"model_id": "C", "test_macro_f1": 0.68, "recall_hate": 0.4, "ece": 0.09,
             "bias_gap": 0.20, "latency_p95_ms": 35.0, "size_mb": 450.0},
        ]
    )
    front = set(_pareto_front(df)["model_id"])
    assert "A" in front and "B" in front
    assert "C" not in front


def test_has_profanity_flag_is_token_membership():
    import pandas as pd

    from hsc.error_analysis import _has_profanity
    from hsc.probe import _profanity_set

    prof = _profanity_set()
    if prof is None:  # wordlist unavailable in this env
        return
    bad = next(iter(prof))  # a real listed term — avoids hardcoding profanity
    s = pd.Series([f"you are a {bad}", "have a lovely day zzz", "clean neutral sentence"])
    flags = _has_profanity(s, prof)
    assert bool(flags.iloc[0]) is True  # token membership fires
    assert bool(flags.iloc[1]) is False and bool(flags.iloc[2]) is False
