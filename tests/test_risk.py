"""Regression tests for stable risk-classifier evaluation output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import risk


def test_classifier_confusion_matrix_keeps_all_risk_classes(monkeypatch):
    class FakeClassifier:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    monkeypatch.setattr(risk, "XGBClassifier", FakeClassifier)
    count = 16
    districts = pd.DataFrame(
        {
            "risk_score": np.repeat(np.arange(4), 4),
            "level_mbgl": np.arange(count, dtype=float),
            "predicted_decline_m_per_year": np.arange(count, dtype=float),
            "trend_slope": np.arange(count, dtype=float),
            "annual_rainfall_mm": np.arange(count, dtype=float),
            "rainfall_rolling3y": np.arange(count, dtype=float),
            "season_flag": np.arange(count, dtype=float),
        }
    )

    _, report, matrix, _ = risk.train_risk_classifier(districts, {})

    assert matrix.shape == (4, 4)
    assert all(label in report for label in risk.LABELS)
