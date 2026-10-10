"""Tests for the synthetic data generator and cleaning rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.aggregate import aggregate_to_district
from src.data.clean import clean_data
from src.data.ingest import ingest_local
from src.data.synthetic import generate_synthetic
from src.config import Config, load_config


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


def test_clean_preserves_distinct_rainfall_rows_and_interpolates_by_district(cfg):
    dates = pd.to_datetime(["2020-05-01", "2020-12-01", "2021-05-01"])
    data = pd.DataFrame(
        {
            "district": ["Alpha"] * 3 + ["Beta"] * 3,
            "station_id": ["shared-station"] * 6,
            "date": list(dates) * 2,
            "level_mbgl": [10.0, np.nan, 14.0, 100.0, np.nan, 104.0],
            "rainfall_mm": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    cleaned = clean_data(data, cfg)

    alpha_middle = cleaned[
        (cleaned["district"] == "Alpha") & cleaned["date"].eq(dates[1])
    ]["level_mbgl"].iloc[0]
    beta_middle = cleaned[
        (cleaned["district"] == "Beta") & cleaned["date"].eq(dates[1])
    ]["level_mbgl"].iloc[0]
    assert len(cleaned) == len(data)
    assert alpha_middle == 12.0
    assert beta_middle == 102.0

    distinct_rainfall = pd.DataFrame(
        {
            "district": ["Alpha", "Alpha"],
            "station_id": ["A-1", "A-1"],
            "date": [dates[0], dates[0]],
            "level_mbgl": [10.0, 10.0],
            "rainfall_mm": [1.0, 9.0],
        }
    )
    preserved = clean_data(distinct_rainfall, cfg)
    assert len(preserved) == 2
    assert set(preserved["rainfall_mm"]) == {1.0, 9.0}


def test_ingest_local_maps_measurements_and_expands_wide_seasons(tmp_path, cfg):
    source = pd.DataFrame(
        {
            "District Name": ["Alpha"],
            "State Name": ["State A"],
            "Station Code": ["A-1"],
            "Year": [2020],
            "Pre-monsoon": [12.5],
            "Post-monsoon": [11.0],
            "Rainfall": [750.0],
        }
    )
    source.to_csv(tmp_path / "wide.csv", index=False)
    config_data = {**cfg.raw, "data": {**cfg.data_cfg, "local_path": str(tmp_path)}}

    ingested = ingest_local(Config(raw=config_data))

    assert len(ingested) == 2
    assert set(ingested["level_mbgl"]) == {12.5, 11.0}
    assert ingested["rainfall_mm"].eq(750.0).all()
    assert set(ingested["season"]) == {"pre-monsoon", "post-monsoon"}
    assert set(ingested["date"].dt.month) == {5, 11}


def test_ingest_combines_state_scoped_groundwater_with_chunked_daily_rainfall(
    tmp_path, cfg
):
    groundwater = pd.DataFrame(
        {
            "district_name": ["Springfield"] * 4,
            "state_name": ["State A", "State A", "State B", "State B"],
            "station_name": ["Well 1", "Well 1", "Well 2", "Well 2"],
            "date": ["2019-05-15", "2019-11-15"] * 2,
            "currentlevel": [10.0, 11.0, 100.0, 101.0],
            "latitude": [10.0, 10.0, 20.0, 20.0],
            "longitude": [30.0, 30.0, 40.0, 40.0],
        }
    )
    rainfall = pd.DataFrame(
        {
            "district_name": ["Springfield"] * 6,
            "state_name": ["State A"] * 3 + ["State B"] * 3,
            "date": [
                "2019-01-01", "2019-05-01", "2019-11-01"
            ] * 2,
            "actual": [10.0, 2.0, 3.0, 20.0, 4.0, 6.0],
        }
    )
    off_season_rainfall = pd.DataFrame(
        {
            "district_name": ["Springfield"] * 2,
            "state_name": ["State A", "State B"],
            "date": ["2019-08-01"] * 2,
            "actual": [5.0, 10.0],
        }
    )
    groundwater.to_csv(tmp_path / "groundwater.csv", index=False)
    rainfall.to_csv(tmp_path / "rainfall.csv", index=False)
    off_season_rainfall.to_csv(tmp_path / "off_season_rainfall.csv", index=False)
    config_data = {**cfg.raw, "data": {**cfg.data_cfg, "local_path": str(tmp_path)}}

    ingested = ingest_local(Config(raw=config_data))
    cleaned = clean_data(ingested, Config(raw=config_data))
    levels, rainfall_aggregates = aggregate_to_district(cleaned)

    levels_by_state = levels.set_index(["state", "season"])["level_mbgl"]
    rain_by_state = rainfall_aggregates.groupby("state")["annual_rainfall_mm"].first()
    assert levels_by_state.loc[("State A", "pre-monsoon")] == 10.0
    assert levels_by_state.loc[("State B", "pre-monsoon")] == 100.0
    assert rain_by_state["State A"] == 20.0
    assert rain_by_state["State B"] == 40.0
    seasonal_rain = rainfall_aggregates.groupby("state")["rainfall_mm"].sum()
    assert seasonal_rain["State A"] == 5.0
    assert seasonal_rain["State B"] == 10.0


def test_aggregate_shapes(cfg, raw):
    clean = clean_data(raw, cfg)
    levels, rainfall = aggregate_to_district(clean)
    assert {"district", "year", "season", "level_mbgl", "date"}.issubset(levels.columns)
    assert set(levels["season"].unique()) <= {"pre-monsoon", "post-monsoon"}
    assert len(levels) == len(levels.drop_duplicates(["district", "year", "season"]))
    if not rainfall.empty:
        assert "annual_rainfall_mm" in rainfall.columns


def test_aggregate_excludes_off_season_from_seasonal_means_and_totals():
    data = pd.DataFrame(
        {
            "district": ["Alpha"] * 3,
            "state": ["State A"] * 3,
            "station_id": ["A-1"] * 3,
            "date": pd.to_datetime(["2020-05-15", "2020-08-15", "2020-11-15"]),
            "level_mbgl": [10.0, 99.0, 12.0],
            "rainfall_mm": [2.0, 5.0, 3.0],
        }
    )

    levels, rainfall = aggregate_to_district(data)

    assert set(levels["season"]) == {"pre-monsoon", "post-monsoon"}
    assert len(levels) == 2
    assert set(rainfall["season"]) == {"pre-monsoon", "post-monsoon"}
    assert rainfall["rainfall_mm"].sum() == 5.0
    assert rainfall["annual_rainfall_mm"].eq(10.0).all()
