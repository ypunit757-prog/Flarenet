import numpy as np
import pytest

from flarenet.pipelines.evaluation.metrics import (
    best_threshold,
    contingency,
    hss,
    per_class_metrics,
    safe_auc,
    safe_roc,
    score_at_threshold,
    tss,
)


def test_contingency_counts():
    assert contingency([1, 1, 0, 0, 1], [1, 0, 0, 1, 1]) == (2, 1, 1, 1)


def test_tss_and_hss_known_values():
    assert tss(10, 0, 0, 90) == pytest.approx(1.0)
    assert tss(0, 10, 10, 80) == pytest.approx(0.0 - 10 / 90)
    assert hss(10, 0, 0, 90) == pytest.approx(1.0)
    assert hss(5, 5, 5, 5) == pytest.approx(0.0)


def test_score_at_threshold_handles_no_positives():
    s = score_at_threshold([0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert np.isnan(s["POD"]) and np.isnan(s["FAR"]) and s["TSS"] == 0.0


def test_best_threshold_on_separable_data():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.05, 0.4, 50), np.linspace(0.6, 0.95, 50)])
    thr, score = best_threshold(y, p)
    assert 0.4 <= thr <= 0.6 and score == pytest.approx(1.0)


def test_auc_and_roc_single_class_fallback():
    assert np.isnan(safe_auc([0, 0, 0], [0.1, 0.2, 0.3]))
    fpr, tpr = safe_roc([1, 1], [0.4, 0.6])
    assert fpr.tolist() == [0.0, 1.0] and tpr.tolist() == [0.0, 1.0]


def test_per_class_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 3, 4, 2, 3])
    rows = per_class_metrics(y, y)
    assert [r["class"] for r in rows] == ["C", "M", "X"]
    assert all(r["TSS"] == pytest.approx(1.0) for r in rows)
