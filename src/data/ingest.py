"""Path-driven ingestion of real station-level data (CSV / Excel).

Reads every CSV/XLSX file found at the configured path (file or folder),
sniffs columns against an alias table, and normalises to the canonical
schema used across the pipeline:

    district, state, station_id, date, level_mbgl, rainfall_mm,
    latitude, longitude
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)

CANONICAL_COLS = [
    "district", "state", "station_id", "date",
    "level_mbgl", "rainfall_mm", "latitude", "longitude",
]


def _find_files(path: Path) -> list[Path]:
    """Return every CSV/XLSX file at `path` (file or folder, recursive)."""
    if path.is_file():
        return [path]
    files: list[Path] = []
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(path.rglob(pattern))
    # ignore temp excel lock files
    return sorted(p for p in files if not p.name.startswith("~$"))


def _map_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rename file columns to canonical names using the alias table."""
    rename: dict[str, str] = {}
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for canonical, aliases in cfg.column_aliases.items():
        for alias in aliases:
            key = str(alias).strip().lower()
            if key in lowered and canonical not in rename.values():
                rename[lowered[key]] = canonical
                break
    return df.rename(columns=rename)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    for col in ("level_mbgl", "rainfall_mm", "latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("district", "state", "station_id"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def _derive_station_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a station_id; synthesise one from district + row order if absent."""
    if "station_id" not in df.columns or df["station_id"].isna().all():
        df = df.copy()
        if "date" in df.columns:
            df = df.sort_values(["district", "date"])
        grp = df.groupby("district", dropna=False).cumcount()
        df["station_id"] = df["district"].str.replace(r"\W+", "_", regex=True) + "_S" + (grp % 3 + 1).astype(str)
    return df


def _seasonal_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Handle CGWB-style files where pre/post monsoon levels sit in separate
    columns on one row per station-year."""
    pre_cols = [c for c in df.columns if "pre" in c.lower() and "monsoon" in c.lower() or c.lower() == "pre-monsoon"]
    post_cols = [c for c in df.columns if "post" in c.lower() and "monsoon" in c.lower() or c.lower() == "post-monsoon"]
    if not (pre_cols or post_cols):
        return df
    long_parts = []
    for cols, season, month in ((pre_cols, "pre-monsoon", 5), (post_cols, "post-monsoon", 11)):
        for col in cols:
            part = df.copy()
            part["level_mbgl"] = pd.to_numeric(part[col], errors="coerce")
            part["date"] = pd.to_datetime(
                dict(year=part["date"].dt.year, month=month, day=15), errors="coerce"
            )
            part["season"] = season
            long_parts.append(part[CANONICAL_COLS + ["season"]])
    out = pd.concat(long_parts, ignore_index=True)
    return out


def ingest_local(cfg: Config) -> pd.DataFrame:
    """Ingest all files under the configured local path into canonical schema."""
    base = Path(cfg.local_path)
    if not base.exists():
        raise FileNotFoundError(f"Configured data path does not exist: {base}")

    frames: list[pd.DataFrame] = []
    for f in _find_files(base):
        logger.info("Reading %s", f)
        try:
            if f.suffix.lower() == ".csv":
                raw = pd.read_csv(f, low_memory=False)
            elif f.suffix.lower() in (".xlsx", ".xls"):
                raw = pd.read_excel(f)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - keep going on a bad file
            logger.warning("Could not read %s: %s", f, exc)
            continue

        raw = _map_columns(raw, cfg)
        if "level_mbgl" in raw.columns and raw["level_mbgl"].isna().all():
            raw = _seasonal_wide_to_long(raw)
        if "date" in raw.columns and np.issubdtype(raw["date"].dtype, np.datetime64):
            pass
        elif "date" in raw.columns:
            raw = _coerce_types(raw)
        else:
            raw = _coerce_types(raw)

        if "district" not in raw.columns:
            logger.warning("%s has no district-like column; skipping", f)
            continue
        raw = _derive_station_id(raw)

        for col in CANONICAL_COLS:
            if col not in raw.columns:
                raw[col] = np.nan
        frames.append(raw[CANONICAL_COLS])

    if not frames:
        raise ValueError(
            f"No readable data files with a district column found under {base}. "
            "Check config.yaml -> data.local_path or drop files into data/raw/."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["district", "date"])
    df["season"] = np.where(df["date"].dt.month.isin([3, 4, 5]), "pre-monsoon", "post-monsoon")
    return df.sort_values(["district", "station_id", "date"]).reset_index(drop=True)


def load_data(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Load station-level data from the configured source.

    Returns (dataframe, source_used) where source_used is
    "local", "synthetic" or "synthetic-fallback".
    """
    if cfg.data_source == "local":
        try:
            return ingest_local(cfg), "local"
        except Exception as exc:  # noqa: BLE001
            if not cfg.allow_synthetic_fallback:
                raise
            logger.warning("Local ingestion failed (%s); falling back to synthetic data.", exc)
            from src.data.synthetic import generate_synthetic
            return generate_synthetic(cfg), "synthetic-fallback"
    from src.data.synthetic import generate_synthetic
    return generate_synthetic(cfg), "synthetic"
