"""Contract test for the product inference wrapper. Skips when no model is trained yet."""

from __future__ import annotations

import pytest

from hsc.config import resolve


def _has_servable_model() -> bool:
    # A locally-loadable classical bundle (joblib), not just a registry entry — neural
    # weights live on Colab and the joblibs are gitignored, so CI skips this contract test.
    return any(resolve("models").glob("*/model.joblib"))


pytestmark = pytest.mark.skipif(
    not _has_servable_model(), reason="no locally-servable model.joblib (e.g. fresh CI checkout)"
)


def test_predict_contract():
    from hsc.inference import HateClassifier

    clf = HateClassifier()
    out = clf.predict("I love everyone in this community")
    assert set(out) >= {"text", "label", "score", "language", "model_version"}
    assert out["label"] in {"hate", "not_hate"}
    assert 0.0 <= out["score"] <= 1.0
    assert out["language"]["detected"] in {"en", "pt", "other"}


def test_predict_batch_and_pt():
    from hsc.inference import HateClassifier

    clf = HateClassifier()
    outs = clf.predict_batch(["hello there", "olá, bom dia a todos"])
    assert len(outs) == 2
    assert all(o["label"] in {"hate", "not_hate"} for o in outs)
    # the Portuguese line should be detected as pt
    assert outs[1]["language"]["detected"] == "pt"
