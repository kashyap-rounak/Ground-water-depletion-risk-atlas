"""Tests for the synthetic data generator and cleaning rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.synthetic import generate_synthetic
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw(cfg):
    return generate_synthetic(cfg, seed=7)


def test_synthetic_schema(raw):
    for col in ("district", "state", "station_id", "date", "level_mbgl", "rainfall_mm"):
        assert col in raw.columns
    assert raw["district"].nunique() >= 30
    assert raw["date"].dt.month.isin([3, 4, 5, 10, 11, 12]).all()


def test_clean_removes_sentinel_bad_values(cfg, raw):
    raw_with_bad = pd.concat([raw, raw.head(20).assign(level_mbgl=-7.0)], ignore_index=True)
    out = clean_data(raw_with_bad, cfg)
    assert (out["level_mbgl"] >= 0).all() or out["level_mbgl"].isna().any()
    assert not (out["level_mbgl"] < 0).any()


def test_clean_dedupes(cfg, raw):
    doubled = pd.concat([raw, raw], ignore_index=True)
    out = clean_data(doubled, cfg)
    assert len(out) == len(raw)


def test_aggregate_shapes(cfg, raw):
    clean = clean_data(raw, cfg)
    levels, rainfall = aggregate_to_district(clean)
    assert {"district", "year", "season", "level_mbgl", "date"}.issubset(levels.columns)
    assert set(levels["season"].unique()) <= {"pre-monsoon", "post-monsoon"}
    assert len(levels) == len(levels.drop_duplicates(["district", "year", "season"]))
    if not rainfall.empty:
        assert "annual_rainfall_mm" in rainfall.columns
