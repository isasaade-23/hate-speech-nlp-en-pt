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
