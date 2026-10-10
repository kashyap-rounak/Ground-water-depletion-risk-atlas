"""Tests for pipeline stage helpers that do not train or persist models."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config
from scripts.run_pipeline import (
    PreparedData,
    _build_district_summary,
    _save_run_metadata,
)


def test_build_district_summary_aligns_latest_features_and_forecast_slopes():
    features = pd.DataFrame(
        {
            "district": ["Alpha", "Alpha", "Beta", "Beta"],
            "date": pd.to_datetime(
                ["2020-05-01", "2020-12-01", "2020-05-01", "2020-12-01"]
            ),
            "level_mbgl": [10.0, 11.0, 20.0, 19.0],
            "annual_rainfall_mm": [100.0, 100.0, 200.0, 200.0],
            "rainfall_rolling3y": [90.0, 95.0, 180.0, 190.0],
            "season_flag": [0, 1, 0, 1],
        }
    )
    decline = pd.DataFrame(
        {
            "district": ["Beta", "Alpha"],
            "state": ["State B", "State A"],
            "decline_m_per_year": [0.4, 0.2],
        }
    )
    future = pd.DataFrame(
        {
            "district": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Beta"],
            "date": pd.to_datetime(
                [
                    "2021-05-01",
                    "2021-12-01",
                    "2022-05-01",
                    "2021-05-01",
                    "2021-12-01",
                    "2022-05-01",
                ]
            ),
            "forecast_level": [12.0, 13.0, 14.0, 20.0, 19.0, 18.0],
        }
    )

    result = _build_district_summary(features, decline, future).set_index("district")

    assert result.loc["Alpha", "level_mbgl"] == 11.0
    assert result.loc["Beta", "annual_rainfall_mm"] == 200.0
    assert result.loc["Alpha", "state"] == "State A"
    assert np.isclose(result.loc["Alpha", "predicted_decline_m_per_year"], 2.0)
    assert np.isclose(result.loc["Beta", "predicted_decline_m_per_year"], -2.0)


def test_run_metadata_records_source_and_date_range(tmp_path):
    dates = pd.to_datetime(["2009-01-01", "2019-05-01", "2021-11-01"])
    prepared = PreparedData(
        clean=pd.DataFrame(
            {
                "date": dates,
                "district": ["Alpha"] * 3,
                "state": ["State A"] * 3,
                "level_mbgl": [np.nan, 12.0, 11.0],
            }
        ),
        levels=pd.DataFrame(),
        rainfall=pd.DataFrame(),
        features=pd.DataFrame(),
        decline=pd.DataFrame(),
        correlation=pd.DataFrame(),
        source_used="local",
    )

    _save_run_metadata(prepared, load_config(), tmp_path)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_used"] == "local"
    assert metadata["rows"] == 3
    assert metadata["groundwater_observations"] == 2
    assert metadata["seasonal_groundwater_observations"] == 2
    assert metadata["groundwater_observations_excluded_by_season"] == 0
    assert metadata["districts"] == 1
    assert metadata["date_min"].startswith("2019-05-01")
    assert metadata["date_max"].startswith("2021-11-01")
