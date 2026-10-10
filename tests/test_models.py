"""Tests for leakage-safe time-series model evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.models import xgboost_model


def _panel_frame(n_dates: int = 24) -> pd.DataFrame:
    records = []
    dates = pd.date_range("2012-05-01", periods=n_dates, freq="6MS")
    for district_index, district in enumerate(("Alpha", "Beta")):
        for date_index, date in enumerate(dates):
            row = {
                "district": district,
                "date": date,
                "target_next_level": float(1000 * district_index + date_index + 1),
            }
            row.update({feature: float(date_index + district_index) for feature in FEATURE_COLUMNS})
            records.append(row)
    return pd.DataFrame(records)


def test_chronological_split_purges_training_targets_at_test_boundary():
    frame = _panel_frame()

    train, test = xgboost_model.chronological_split(frame, test_seasons=4)

    next_train_dates = xgboost_model._next_observation_dates(train)
    assert train["date"].max() < test["date"].min()
    assert next_train_dates.max() < test["date"].min()
    assert test["date"].nunique() == 4


def test_walk_forward_folds_use_dates_and_purge_cross_boundary_targets():
    frame = _panel_frame()

    folds = list(xgboost_model._walk_forward_splits(frame, n_splits=4))

    assert len(folds) == 4
    for train, validation in folds:
        next_train_dates = xgboost_model._next_observation_dates(train)
        assert train["date"].max() < validation["date"].min()
        assert next_train_dates.max() < validation["date"].min()


def test_holdout_is_not_used_for_early_stopping(monkeypatch):
    frame = _panel_frame()
    fit_calls: list[dict] = []
    model_configs: list[dict] = []

    class FakeRegressor:
        def __init__(self, *, early_stopping: bool, config: dict):
            self.early_stopping = early_stopping
            self.config = config
            self.best_iteration = 2
            model_configs.append(config)

        def fit(self, X, y, eval_set=None, verbose=None):
            fit_calls.append(
                {
                    "early_stopping": self.early_stopping,
                    "targets": np.asarray(y),
                    "eval_targets": (
                        np.asarray(eval_set[0][1]) if eval_set is not None else None
                    ),
                }
            )
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    def fake_make_model(config: dict, early_stopping: bool = False):
        return FakeRegressor(early_stopping=early_stopping, config=config)

    monkeypatch.setattr(xgboost_model, "_make_model", fake_make_model)
    config = {"test_seasons": 4, "n_estimators": 600, "early_stopping_rounds": 50}

    model, metrics, predictions, test = xgboost_model.train_xgb_regressor(frame, config)

    test_targets = set(test["target_next_level"])
    assert len(fit_calls) == 2
    assert fit_calls[0]["early_stopping"]
    assert not fit_calls[1]["early_stopping"]
    assert test_targets.isdisjoint(fit_calls[0]["eval_targets"])
    assert test_targets.isdisjoint(fit_calls[0]["targets"])
    assert model.config["n_estimators"] == 3
    assert len(predictions) == len(test)
    assert "baseline_persistence_mae" in metrics
