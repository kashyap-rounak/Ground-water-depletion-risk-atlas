"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.synthetic import generate_synthetic
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district
from src.features.build_features import (
    build_features,
    drop_rows_without_target,
    FEATURE_COLUMNS,
)


# Fixture to generate and prepare features for testing.
@pytest.fixture(scope="module")
def feats():
    # Load the project configuration.
    cfg = load_config()

    # Generate synthetic groundwater and rainfall data using a fixed seed.
    raw = generate_synthetic(cfg, seed=7)

    # Clean the raw dataset by removing invalid values and duplicates.
    clean = clean_data(raw, cfg)

    # Aggregate station-level data into district-level groundwater and rainfall data.
    levels, rainfall = aggregate_to_district(clean)

    # Build features such as lag values and the next groundwater-level target.
    return build_features(levels, rainfall)


# Test whether all required feature columns are present in the dataset.
def test_feature_columns_present(feats):
    # Check each required feature column.
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing {col}"


# Test whether the lag-1 feature contains the previous groundwater-level value.
def test_lags_are_shifted(feats):
    # Select the first district and sort its records chronologically.
    d = feats[
        feats["district"] == feats["district"].iloc[0]
    ].sort_values("date")

    # Verify that the second row's lag-1 value equals the first row's groundwater level.
    assert np.isclose(d["level_lag1"].iloc[1], d["level_mbgl"].iloc[0])


# Test whether the target column contains the next groundwater-level value.
def test_targets_present(feats):
    # Select the first district and sort its records chronologically.
    d = feats[
        feats["district"] == feats["district"].iloc[0]
    ].sort_values("date")

    # Verify that the first row's target equals the second row's groundwater level.
    assert np.isclose(d["target_next_level"].iloc[0], d["level_mbgl"].iloc[1])


# Test whether rows without a target value