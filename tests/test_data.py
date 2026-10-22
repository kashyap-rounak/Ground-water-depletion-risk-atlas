"""Tests for the synthetic data generator and cleaning rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.synthetic import generate_synthetic
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district


# Fixture to load the project configuration once for the entire test module.
@pytest.fixture(scope="module")
def cfg():
    return load_config()


# Fixture to generate synthetic groundwater and rainfall data using a fixed seed.
@pytest.fixture(scope="module")
def raw(cfg):
    return generate_synthetic(cfg, seed=7)


# Test whether the synthetic dataset contains the required columns and valid dates.
def test_synthetic_schema(raw):
    # Check that all required columns exist in the dataset.
    for col in ("district", "state", "station_id", "date", "level_mbgl", "rainfall_mm"):
        assert col in raw.columns

    # Verify that the dataset contains at least 30 unique districts.
    assert raw["district"].nunique() >= 30

    # Check that all dates belong to the pre-monsoon or post-monsoon seasons.
    assert raw["date"].dt.month.isin([3, 4, 5, 10, 11, 12]).all()


# Test whether the cleaning function removes invalid negative groundwater values.
def test_clean_removes_sentinel_bad_values(cfg, raw):
    # Add 20 rows containing an invalid groundwater level of -7.0.
    raw_with_bad = pd.concat(
        [raw, raw.head(20).assign(level_mbgl=-7.0)],
        ignore_index=True
    )

    # Clean the dataset by applying the configured cleaning rules.
    out = clean_data(raw_with_bad, cfg)

    # Verify that no negative groundwater levels remain.
    assert (out["level_mbgl"] >= 0).all() or out["level_mbgl"].isna().any()
    assert not (out["level_mbgl"] < 0).any()


# Test whether duplicate records are removed during data cleaning.
def test_clean_dedupes(cfg, raw):
    # Duplicate the entire dataset by concatenating it with itself.
    doubled = pd.concat([raw, raw], ignore_index=True)

    # Apply the data-cleaning function to the duplicated dataset.
    out = clean_data(doubled, cfg)

    # Verify that the cleaned dataset has the original number of rows.
    assert len(out) == len(raw)


# Test whether cleaned station data is correctly aggregated to district level.
def test_aggregate_shapes(cfg, raw):
    # Clean the raw synthetic dataset before aggregation.
    clean = clean_data(raw, cfg)

    # Aggregate groundwater levels and rainfall by district and season.
    levels, rainfall = aggregate_to_district(clean)

    # Check that the aggregated groundwater dataset has the required columns.
    assert {"district", "year", "season", "level_mbgl", "date"}.issubset(levels.columns)

    # Verify that only the two expected seasons appear in the results.
    assert set(levels["season"].unique()) <= {"pre-monsoon", "post-monsoon"}

    # Ensure each district has at most one record per year and season.
    assert len(levels) == len(
        levels.drop_duplicates(["district", "year", "season"])
    )

    # If rainfall data exists, verify that annual rainfall is included.
    if not rainfall.empty:
        assert "annual_rainfall_mm" in rainfall.columns