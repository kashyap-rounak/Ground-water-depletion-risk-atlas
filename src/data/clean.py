"""Data cleaning for station-level groundwater and rainfall readings.

Steps (each logged with row counts):
  1. drop exact duplicates
  2. type coercion already done in ingest; here we sanity-filter values
  3. IQR-based outlier flagging within each district's series
  4. time-interpolation of missing levels within each station's series
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

CANONICAL_COLS = [
    "district", "state", "station_id", "date",
    "level_mbgl", "rainfall_mm", "latitude", "longitude",
]


def clean_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the configured cleaning rules. Returns a cleaned copy."""
    required = {"district", "station_id", "date", "level_mbgl", "rainfall_mm"}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required cleaning columns: {', '.join(missing_columns)}")

    cl = cfg.cleaning
    n0 = len(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("level_mbgl", "rainfall_mm", "annual_rainfall_mm"):
        if column not in out.columns:
            continue
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("district", "station_id"):
        values = out[column].astype("string").str.strip()
        out[column] = values.mask(values.str.lower().isin(["", "nan", "none", "null"]))

    invalid_key = out["district"].isna() | out["date"].isna()
    n_invalid_key = int(invalid_key.sum())
    out = out.loc[~invalid_key].copy()

    # --- 1. exact duplicate rows -------------------------------------
    rows_before_deduplication = len(out)
    out = out.drop_duplicates(keep="first")
    n_dupes = rows_before_deduplication - len(out)

    # --- 2. sanity ranges ---------------------------------------------
    lvl_min = float(cl.get("level_min_mbgl", 0.0))
    lvl_max = float(cl.get("level_max_mbgl", 120.0))
    bad_level = out["level_mbgl"].notna() & (
        (out["level_mbgl"] < lvl_min) | (out["level_mbgl"] > lvl_max)
    )
    rain_min = float(cl.get("rainfall_min_mm", 0.0))
    rain_max = float(cl.get("rainfall_max_mm", 5000.0))
    bad_rain = out["rainfall_mm"].notna() & (
        (out["rainfall_mm"] < rain_min) | (out["rainfall_mm"] > rain_max)
    )
    out.loc[bad_level, "level_mbgl"] = np.nan
    nullified_level = int(bad_level.sum())
    out.loc[bad_rain, "rainfall_mm"] = np.nan

    # --- 3. IQR outlier fence within district -------------------------
    mult = float(cl.get("iqr_multiplier", 3.0))
    def _iqr_mask(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 8:
            return pd.Series(False, index=s.index)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - mult * iqr) | (s > q3 + mult * iqr)
    entity_keys = (
        ["state", "district"]
        if "state" in out.columns and out["state"].notna().any()
        else ["district"]
    )
    iqr_mask = out.groupby(entity_keys, dropna=False)["level_mbgl"].transform(_iqr_mask)
    out.loc[iqr_mask, "level_mbgl"] = np.nan
    n_outliers = int(iqr_mask.sum())

    # --- 4. time interpolation within each station ---------------------
    if cl.get("interpolate", True):
        out = out.sort_values(["district", "station_id", "date"])
        out["level_mbgl"] = (
            out.groupby([*entity_keys, "station_id"], dropna=False)["level_mbgl"]
            .transform(lambda s: s.interpolate(method="linear", limit=4, limit_area="inside"))
        )
        out = out.sort_values(["district", "station_id", "date"]).reset_index(drop=True)

    logger.info(
        "Cleaning: %d rows in -> %d out | invalid_keys=%d dupes=%d "
        "bad_level=%d bad_rain=%d outliers=%d missing_levels=%d missing_rainfall=%d",
        n0,
        len(out),
        n_invalid_key,
        n_dupes,
        nullified_level,
        int(bad_rain.sum()),
        n_outliers,
        int(out["level_mbgl"].isna().sum()),
        int(out["rainfall_mm"].isna().sum()),
    )
    return out
