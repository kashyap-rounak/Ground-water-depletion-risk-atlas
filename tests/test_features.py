"""Tests for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.synthetic import generate_synthetic
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district
from src.features.build_features import build_features, drop_rows_without_target, FEATURE_COLUMNS


@pytest.fixture(scope="module")
def feats():
    cfg = load_config()
    raw = generate_synthetic(cfg, seed=7)
    clean = clean_data(raw, cfg)
    levels, rainfall = aggregate_to_district(clean)
    return build_features(levels, rainfall)


def test_feature_columns_present(feats):
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing {col}"


def test_lags_are_shifted(feats):
    d = feats[feats["district"] == feats["district"].iloc[0]].sort_values("date")
    assert np.isclose(d["level_lag1"].iloc[1], d["level_mbgl"].iloc[0])


def test_targets_present(feats):
    d = feats[feats["district"] == feats["district"].iloc[0]].sort_values("date")
    assert np.isclose(d["target_next_level"].iloc[0], d["level_mbgl"].iloc[1])


def test_supervised_rows(feats):
    sup = drop_rows_without_target(feats)
    assert sup["target_next_level"].notna().all()
    assert len(sup) < len(feats)


def test_identically_named_district_series_are_scoped_by_state():
    levels = pd.DataFrame(
        {
            "district": ["Springfield"] * 4,
            "state": ["State A"] * 2 + ["State B"] * 2,
            "year": [2020, 2020, 2020, 2020],
            "season": ["pre-monsoon", "post-monsoon"] * 2,
            "date": pd.to_datetime(
                ["2020-05-01", "2020-12-01"] * 2
            ),
            "level_mbgl": [10.0, 11.0, 100.0, 101.0],
        }
    )
    rainfall = pd.DataFrame(
        {
            "district": ["Springfield"] * 4,
            "state": ["State A"] * 2 + ["State B"] * 2,
            "year": [2020] * 4,
            "season": ["pre-monsoon", "post-monsoon"] * 2,
            "rainfall_mm": [2.0, 3.0, 20.0, 30.0],
            "annual_rainfall_mm": [5.0, 5.0, 50.0, 50.0],
        }
    )

    features = build_features(levels, rainfall).set_index(["state", "date"])

    assert pd.isna(features.loc[("State B", pd.Timestamp("2020-05-01")), "level_lag1"])
    assert features.loc[("State B", pd.Timestamp("2020-12-01")), "level_lag1"] == 100.0
    assert features.loc[("State B", pd.Timestamp("2020-05-01")), "annual_rainfall_mm"] == 50.0
