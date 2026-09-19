"""Feature engineering on the district-season master table.

Creates, per district (ordered by season end-date):
  - level_lag1 / lag2 / lag4 (previous seasons' groundwater depth)
  - level_change_yoy (same-season, one year earlier)
  - rainfall_rolling3y (mean annual rainfall over trailing 3 years)
  - rainfall_lag1_season / rainfall_lag1_year
  - season flag and cyclical seasonal encoding
  - trend_slope (m/yr over trailing 5 seasons, trailing-window OLS)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.aggregate import SEASON_ORDER

FEATURE_COLUMNS = [
    "level_lag1", "level_lag2", "level_lag4", "level_change_yoy",
    "rainfall_rolling3y", "rainfall_lag1_season", "rainfall_lag1_year",
    "season_flag", "season_sin", "season_cos", "trend_slope",
]


def build_features(
    levels: pd.DataFrame,
    rainfall: pd.DataFrame,
    rolling_years: int = 3,
    yoy_seasons: int = 2,
) -> pd.DataFrame:
    """Return the master feature table (one row per district-season)."""
    lvl = levels.copy()

    # ----- join annual + seasonal rainfall onto the levels table ----------
    ann = (
        rainfall.groupby(["district", "year"], as_index=False)["annual_rainfall_mm"]
        .first()
    )
    lvl = lvl.merge(ann, on=["district", "year"], how="left")

    # ----- per-district seasonal lags (ordered by date) --------------------
    lvl = lvl.sort_values(["district", "date"]).copy()
    g = lvl.groupby("district")["level_mbgl"]
    for lag in (1, 2, 4):
        lvl[f"level_lag{lag}"] = g.shift(lag)
    lvl["level_change_yoy"] = lvl["level_mbgl"] - g.shift(yoy_seasons)
    # NOTE: for the regression target we also keep the *next* level
    lvl["target_next_level"] = g.shift(-1)
    lvl["target_next_change"] = lvl["target_next_level"] - lvl["level_mbgl"]

    # ----- rainfall features ------------------------------------------------
    lvl = lvl.merge(
        rainfall[["district", "year", "season", "rainfall_mm"]],
        on=["district", "year", "season"], how="left",
    ).rename(columns={"rainfall_mm": "season_rainfall"})
    lvl["rainfall_rolling3y"] = (
        lvl.groupby("district")["annual_rainfall_mm"].transform(
            lambda s: s.shift(1).rolling(rolling_years, min_periods=1).mean()
        )
    )
    lvl["rainfall_lag1_season"] = lvl.groupby("district")["season_rainfall"].shift(1)
    lvl["rainfall_lag1_year"] = lvl.groupby("district")["annual_rainfall_mm"].shift(2)

    # ----- seasonal encoding -------------------------------------------------
    lvl["season_flag"] = (lvl["season"] == "post-monsoon").astype(int)
    lvl["season_sin"] = np.sin(2 * np.pi * lvl["season_flag"])
    lvl["season_cos"] = np.cos(2 * np.pi * lvl["season_flag"])

    # ----- trailing 5-season OLS slope of level (m per season) ---------------
    def _slope(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) < 3:
            return np.nan
        x = np.arange(len(s), dtype=float)
        slope = np.polyfit(x, s.values, 1)[0]
        return float(slope)

    lvl["trend_slope"] = (
        lvl.groupby("district")["level_mbgl"]
        .transform(lambda s: s.rolling(5, min_periods=3).apply(_slope, raw=False))
    )
    lvl["trend_slope"] = lvl["trend_slope"] / 2.0  # per season -> per year approx

    lvl = lvl.sort_values(["district", "date"]).reset_index(drop=True)
    return lvl


def drop_rows_without_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rows used for supervised training (must have next-season level)."""
    return df.dropna(subset=["target_next_level"]).copy()
