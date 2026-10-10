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
_DEFAULT_CANONICAL = {
    "district": "district",
    "state": "state",
    "station_id": "station_id",
    "date": "date",
    "level": "level_mbgl",
    "rainfall": "rainfall_mm",
    "latitude": "latitude",
    "longitude": "longitude",
}


def _is_seasonal_level_column(name: str) -> bool:
    """Return whether a column represents a wide-format seasonal level."""
    lowered = name.strip().lower()
    return (
        ("pre" in lowered and "monsoon" in lowered)
        or ("post" in lowered and "monsoon" in lowered)
    )


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
        target = cfg.canonical.get(canonical, _DEFAULT_CANONICAL.get(canonical, canonical))
        for alias in aliases:
            key = str(alias).strip().lower()
            if key in lowered and target not in rename.values():
                source = lowered[key]
                if canonical == "level" and _is_seasonal_level_column(str(source)):
                    continue
                rename[source] = target
                break
    return df.rename(columns=rename)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        date_values = out["date"]
        if pd.api.types.is_numeric_dtype(date_values):
            numeric_dates = pd.to_numeric(date_values, errors="coerce")
            observed_dates = numeric_dates.dropna()
            is_year = (
                observed_dates.between(1000, 9999).all()
                and observed_dates.mod(1).eq(0).all()
            )
            if is_year:
                date_values = numeric_dates.astype("Int64").astype("string") + "-01-01"
                out["date"] = pd.to_datetime(date_values, errors="coerce", yearfirst=True)
            else:
                out["date"] = pd.to_datetime(date_values, errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(date_values):
            out["date"] = pd.to_datetime(date_values, errors="coerce")
        else:
            date_text = date_values.astype("string").str.strip()
            iso_dates = date_text.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T].*)?$")
            parsed_dates = pd.Series(pd.NaT, index=date_values.index, dtype="datetime64[ns]")
            parsed_dates.loc[iso_dates] = pd.to_datetime(
                date_text.loc[iso_dates], errors="coerce", yearfirst=True
            )
            parsed_dates.loc[~iso_dates] = pd.to_datetime(
                date_text.loc[~iso_dates], errors="coerce", dayfirst=True
            )
            out["date"] = parsed_dates
    for col in ("level_mbgl", "rainfall_mm", "latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("district", "state", "station_id"):
        if col in out.columns:
            values = out[col].astype("string").str.strip()
            out[col] = values.mask(values.str.lower().isin(["", "nan", "none", "null"]))
    return out


def _derive_station_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a station identifier without splitting a site by row order."""
    if "station_id" not in df.columns or df["station_id"].isna().all():
        df = df.copy()
        district_key = df["district"].astype("string").str.replace(r"\W+", "_", regex=True)
        if "latitude" in df and "longitude" in df:
            location = (
                pd.to_numeric(df["latitude"], errors="coerce").round(5).astype("string")
                + "_"
                + pd.to_numeric(df["longitude"], errors="coerce").round(5).astype("string")
            )
            station_name = df.get("station_name", pd.Series("", index=df.index)).astype("string")
            df["station_id"] = district_key + "_" + station_name.str.replace(
                r"\W+", "_", regex=True
            ) + "_" + location
        else:
            df["station_id"] = district_key + "_S1"
    return df


def _seasonal_wide_to_long(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Handle CGWB-style files where pre/post monsoon levels sit in separate
    columns on one row per station-year."""
    pre_cols = [
        col for col in df.columns
        if "pre" in str(col).lower() and "monsoon" in str(col).lower()
    ]
    post_cols = [
        col for col in df.columns
        if "post" in str(col).lower() and "monsoon" in str(col).lower()
    ]
    if not (pre_cols or post_cols):
        return df

    date_column = cfg.canonical.get("date", "date")
    level_column = cfg.canonical.get("level", "level_mbgl")
    if date_column not in df:
        raise ValueError("Seasonal wide-format data requires a date or year column.")

    long_parts = []
    for cols, season, month in ((pre_cols, "pre-monsoon", 5), (post_cols, "post-monsoon", 11)):
        for col in cols:
            part = df.copy()
            part[level_column] = pd.to_numeric(part[col], errors="coerce")
            part[date_column] = pd.to_datetime(
                dict(year=part[date_column].dt.year, month=month, day=15), errors="coerce"
            )
            part["season"] = season
            long_parts.append(part.drop(columns=[*pre_cols, *post_cols]))
    if not long_parts:
        return df
    out = pd.concat(long_parts, ignore_index=True)
    return out


def _aggregate_daily_rainfall(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Reduce district-level daily rainfall to seasonal rows before concatenation."""
    required = {"district", "state", "date", "rainfall_mm"}
    if not required.issubset(df.columns):
        raise ValueError(
            "District rainfall input requires district, state, date, and rainfall columns."
        )
    data = df.dropna(subset=["district", "state", "date", "rainfall_mm"]).copy()
    if data.empty:
        return pd.DataFrame(columns=[*CANONICAL_COLS, "annual_rainfall_mm", "season"])

    data["year"] = data["date"].dt.year
    pre_months = tuple(cfg.seasons.get("pre_monsoon_months", [3, 4, 5]))
    post_months = tuple(cfg.seasons.get("post_monsoon_months", [10, 11, 12]))
    month = data["date"].dt.month
    data["season"] = np.select(
        [month.isin(pre_months), month.isin(post_months)],
        ["pre-monsoon", "post-monsoon"],
        default="other",
    )
    entity_year = ["state", "district", "year"]
    annual = (
        data.groupby(entity_year, as_index=False)
        .agg(annual_rainfall_mm=("rainfall_mm", lambda values: values.sum(min_count=1)))
    )
    seasonal = (
        data[data["season"].ne("other")]
        .groupby([*entity_year, "season"], as_index=False)
        .agg(rainfall_mm=("rainfall_mm", lambda values: values.sum(min_count=1)))
        .merge(annual, on=entity_year, how="left")
    )
    seasonal["date"] = pd.to_datetime(
        {
            "year": seasonal["year"],
            "month": np.where(seasonal["season"].eq("pre-monsoon"), 5, 12),
            "day": 1,
        }
    )
    seasonal["station_id"] = (
        seasonal["state"].astype("string").str.replace(r"\W+", "_", regex=True)
        + "_"
        + seasonal["district"].astype("string").str.replace(r"\W+", "_", regex=True)
        + "_rainfall"
    )
    annual_rows = annual.copy()
    annual_rows["season"] = "other"
    annual_rows["rainfall_mm"] = 0.0
    annual_rows["date"] = pd.to_datetime(
        {"year": annual_rows["year"], "month": 1, "day": 1}
    )
    annual_rows["station_id"] = (
        annual_rows["state"].astype("string").str.replace(r"\W+", "_", regex=True)
        + "_"
        + annual_rows["district"].astype("string").str.replace(r"\W+", "_", regex=True)
        + "_rainfall"
    )
    seasonal["annual_rainfall_mm"] = np.nan
    for frame in (seasonal, annual_rows):
        for column in ("level_mbgl", "latitude", "longitude"):
            frame[column] = np.nan
    seasonal["season"] = seasonal["season"].astype("string")
    annual_rows["season"] = annual_rows["season"].astype("string")
    return pd.concat([seasonal, annual_rows], ignore_index=True)[
        [*CANONICAL_COLS, "annual_rainfall_mm", "season"]
    ]


def _canonicalize_chunk(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Map and normalize one source-data chunk to the pipeline schema."""
    data = _seasonal_wide_to_long(_coerce_types(_map_columns(raw, cfg)), cfg)
    if "district" not in data.columns:
        return data
    data = _derive_station_id(data)
    for column in CANONICAL_COLS:
        if column not in data.columns:
            data[column] = np.nan
    return data


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
                chunks = pd.read_csv(f, low_memory=False, chunksize=100_000)
                file_frames: list[pd.DataFrame] = []
                is_rainfall_only: bool | None = None
                for raw_chunk in chunks:
                    mapped = _coerce_types(_map_columns(raw_chunk, cfg))
                    if "district" not in mapped.columns:
                        logger.warning("%s has no district-like column; skipping", f)
                        file_frames = []
                        break
                    if is_rainfall_only is None:
                        is_rainfall_only = (
                            "level_mbgl" not in mapped.columns
                            and "rainfall_mm" in mapped.columns
                            and not any(
                                _is_seasonal_level_column(str(column))
                                for column in raw_chunk.columns
                            )
                        )
                    if is_rainfall_only:
                        file_frames.append(_aggregate_daily_rainfall(mapped, cfg))
                    else:
                        file_frames.append(_canonicalize_chunk(raw_chunk, cfg))
                frames.extend(file_frames)
                continue
            elif f.suffix.lower() in (".xlsx", ".xls"):
                raw = pd.read_excel(f)
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - keep going on a bad file
            logger.warning("Could not read %s: %s", f, exc)
            continue

        raw = _canonicalize_chunk(raw, cfg)
        if "district" not in raw.columns:
            logger.warning("%s has no district-like column; skipping", f)
            continue
        frames.append(raw)

    if not frames:
        raise ValueError(
            f"No readable data files with a district column found under {base}. "
            "Check config.yaml -> data.local_path or drop files into data/raw/."
        )
    df = pd.concat(frames, ignore_index=True)
    if "annual_rainfall_mm" not in df.columns:
        df["annual_rainfall_mm"] = np.nan
    before_filter = len(df)
    df = df.dropna(subset=["district", "date"])
    dropped = before_filter - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing district or invalid date.", dropped)
    df["district"] = df["district"].astype("string").str.strip()
    df = df[df["district"].notna() & df["district"].ne("")]
    if df.empty:
        raise ValueError(f"No valid rows with district and date found under {base}.")
    if df["level_mbgl"].notna().sum() == 0:
        raise ValueError(f"No valid groundwater-level observations found under {base}.")
    if df["rainfall_mm"].notna().sum() == 0:
        logger.warning("No valid rainfall observations found under %s.", base)

    pre_months = tuple(cfg.seasons.get("pre_monsoon_months", [3, 4, 5]))
    post_months = tuple(cfg.seasons.get("post_monsoon_months", [10, 11, 12]))
    df["season"] = np.select(
        [df["date"].dt.month.isin(pre_months), df["date"].dt.month.isin(post_months)],
        ["pre-monsoon", "post-monsoon"],
        default="other",
    )
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
