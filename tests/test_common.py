import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "churn"))

from common import evaluate, find_threshold, scale_pos_weight  # noqa: E402
from config import SEGMENTS, XGB_PARAMS  # noqa: E402


def test_threshold_separates_a_clean_split():
    y = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([np.linspace(0.0, 0.3, 50), np.linspace(0.7, 1.0, 50)])
    t = find_threshold(y, proba)
    assert 0.3 < t <= 0.7


def test_scale_pos_weight_matches_class_ratio():
    y = np.array([0] * 90 + [1] * 10)
    assert scale_pos_weight(y) == 9.0


def test_evaluate_returns_every_metric_in_range():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = evaluate(y, (proba >= 0.5).astype(int), proba)
    assert set(metrics) == {"AUC", "Accuracy", "Precision", "Recall", "Balanced Accuracy", "F1 Score"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_segment_settings_are_consistent():
    for name, s in SEGMENTS.items():
        assert s["file"].endswith(".parquet"), name
        assert 0.0 < s["threshold"] < 1.0, name
        assert not set(s["actionable"]) & set(s["drop"]), name
        assert not set(s["fill_mode"]) & set(s["fill_median"]), name
    assert XGB_PARAMS["objective"] == "binary:logistic"
