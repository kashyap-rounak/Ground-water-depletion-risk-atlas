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
    required = {"district", "state", "station_id", "date", "level_mbgl", "rainfall_mm"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required aggregation columns: {', '.join(missing_columns)}")
    if any(month < 1 or month > 12 for month in (*pre_months, *post_months)):
        raise ValueError("Season months must be integers from 1 through 12.")
    if set(pre_months).intersection(post_months):
        raise ValueError("Pre- and post-monsoon month sets must not overlap.")

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for column in ("level_mbgl", "rainfall_mm"):
        d[column] = pd.to_numeric(d[column], errors="coerce")
    invalid_date = d["date"].isna()
    if invalid_date.any():
        logger.warning("Ignoring %d rows with invalid dates during aggregation.", int(invalid_date.sum()))
        d = d.loc[~invalid_date].copy()
    d["year"] = d["date"].dt.year
    month = d["date"].dt.month
    d["season"] = np.select(
        [month.isin(pre_months), month.isin(post_months)],
        ["pre-monsoon", "post-monsoon"],
        default="other",
    )
    excluded_level_dates = int(
        (d["level_mbgl"].notna() & d["season"].eq("other")).sum()
    )
    if excluded_level_dates:
        logger.info(
            "Excluding %d groundwater readings outside configured pre/post-monsoon windows.",
            excluded_level_dates,
        )

    # ----- groundwater levels: mean across stations per district-season -----
    g = d[d["level_mbgl"].notna() & d["season"].ne("other")].copy()
    levels = (
        g.groupby(
            ["district", "state", "year", "season"],
            as_index=False,
            dropna=False,
        )
        .agg(level_mbgl=("level_mbgl", "mean"), n_stations=("station_id", "nunique"))
    )
    levels["date"] = [_season_end_date(y, s) for y, s in zip(levels["year"], levels["season"])]
    levels = levels.sort_values(["district", "date"]).reset_index(drop=True)

    # ----- rainfall: seasonal totals + annual totals ------------------------
    r = d[d["rainfall_mm"].notna()].copy()
    if r.empty:
        rainfall = pd.DataFrame(
            columns=[
                "district", "state", "year", "season", "rainfall_mm",
                "annual_rainfall_mm",
            ]
        )
        return levels, rainfall

    rainfall_keys = ["district", "state", "year"]
    if "annual_rainfall_mm" in r and r["annual_rainfall_mm"].notna().any():
        annual = (
            r.dropna(subset=["annual_rainfall_mm"])
            .groupby(rainfall_keys, as_index=False, dropna=False)
            .agg(
                annual_rainfall_mm=(
                    "annual_rainfall_mm",
                    lambda values: values.sum(min_count=1),
                )
            )
        )
    else:
        annual = (
            r.groupby(rainfall_keys, as_index=False, dropna=False)
            .agg(annual_rainfall_mm=("rainfall_mm", "sum"))
        )
    seasonal = (
        r[r["season"].ne("other")]
        .groupby([*rainfall_keys, "season"], as_index=False, dropna=False)
        .agg(seasonal_rainfall_mm=("rainfall_mm", "sum"))
    )
    rainfall = seasonal.merge(annual, on=rainfall_keys, how="left")
    rainfall = rainfall.rename(columns={"seasonal_rainfall_mm": "rainfall_mm"})
    rainfall = rainfall.sort_values(["district", "year", "season"]).reset_index(drop=True)
    logger.info(
        "Aggregated: %d district-seasons, %d rainfall rows, %d districts",
        len(levels), len(rainfall), levels["district"].nunique(),
    )
    return levels, rainfall
