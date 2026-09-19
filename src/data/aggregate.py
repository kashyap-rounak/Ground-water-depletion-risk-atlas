"""Aggregate station-level readings up to district × season resolution.

Produces two tidy tables:
  - district seasons (master): district, state, date(season-end), season,
    level_mbgl (mean across stations), n_stations
  - district rainfall: district, year, season, rainfall_mm (seasonal total),
    plus annual totals
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEASON_ORDER = ["pre-monsoon", "post-monsoon"]


def _season_end_date(row_year: int, season: str) -> pd.Timestamp:
    month = 5 if season == "pre-monsoon" else 12
    return pd.Timestamp(year=row_year, month=month, day=1)


def aggregate_to_district(
    df: pd.DataFrame,
    pre_months: tuple[int, ...] = (3, 4, 5),
    post_months: tuple[int, ...] = (10, 11, 12),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Station rows -> (district_season_levels, district_rainfall)."""
    d = df.copy()
    d["year"] = d["date"].dt.year

    # ----- groundwater levels: mean across stations per district-season -----
    g = d[d["level_mbgl"].notna()].copy()
    g["season"] = np.where(g["date"].dt.month.isin(pre_months), "pre-monsoon", "post-monsoon")
    levels = (
        g.groupby(["district", "state", "year", "season"], as_index=False)
        .agg(level_mbgl=("level_mbgl", "mean"), n_stations=("station_id", "nunique"))
    )
    levels["date"] = [_season_end_date(y, s) for y, s in zip(levels["year"], levels["season"])]
    levels = levels.sort_values(["district", "year", "season"]).reset_index(drop=True)

    # ----- rainfall: seasonal totals + annual totals ------------------------
    r = d[d["rainfall_mm"].notna()].copy()
    if r.empty:
        rainfall = pd.DataFrame(
            columns=["district", "year", "season", "rainfall_mm", "annual_rainfall_mm"]
        )
        return levels, rainfall

    r["season"] = np.where(r["date"].dt.month.isin(pre_months), "pre-monsoon", "post-monsoon")
    seasonal = (
        r.groupby(["district", "year", "season"], as_index=False)
        .agg(seasonal_rainfall_mm=("rainfall_mm", "sum"))
    )
    annual = (
        r.groupby(["district", "year"], as_index=False)
        .agg(annual_rainfall_mm=("rainfall_mm", "sum"))
    )
    rainfall = seasonal.merge(annual, on=["district", "year"], how="left")
    rainfall = rainfall.rename(columns={"seasonal_rainfall_mm": "rainfall_mm"})
    logger.info(
        "Aggregated: %d district-seasons, %d rainfall rows, %d districts",
        len(levels), len(rainfall), levels["district"].nunique(),
    )
    return levels, rainfall
