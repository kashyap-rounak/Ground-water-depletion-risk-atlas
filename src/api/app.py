"""Read-only REST API for the district-level atlas artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pyarrow.lib import ArrowException

from src.config import load_config
from src.risk_categories import LABELS

app = FastAPI(
    title="Groundwater Depletion Risk Atlas API",
    version="1.0.0",
    description=(
        "Read-only access to published district summaries and historical "
        "district-season observations. Station-level data is not exposed."
    ),
)


class DataUnavailableError(RuntimeError):
    """Raised when required published artifacts are not present or readable."""


def _artifact_path(name: str) -> Path:
    cfg = load_config()
    if name == "district_levels.parquet":
        return cfg.processed_dir / name
    if name in {"district_summary.parquet", "atlas_manifest.json"}:
        return cfg.artifacts_dir / name
    raise ValueError(f"Unsupported public artifact: {name}")


def _read_frame(name: str) -> pd.DataFrame:
    path = _artifact_path(name)
    if not path.is_file():
        raise DataUnavailableError(f"Published artifact is missing: {name}")
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported table format: {path.suffix}")
    except (OSError, ValueError, ImportError, ArrowException) as exc:
        raise DataUnavailableError(f"Could not read published artifact: {name}") from exc


def _read_manifest() -> dict[str, Any]:
    path = _artifact_path("atlas_manifest.json")
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataUnavailableError("Could not read atlas run manifest") from exc
    if not isinstance(manifest, dict):
        raise DataUnavailableError("Atlas run manifest must be a JSON object.")
    return manifest


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    safe = frame.astype(object).where(pd.notna(frame), None)
    records = safe.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                record[key] = pd.Timestamp(value).isoformat()
            elif isinstance(value, np.generic):
                record[key] = value.item()
    return records


def _public_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        summary = _read_frame("district_summary.parquet")
        levels = _read_frame("district_levels.parquet")
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    required_summary = {"district", "state", "risk_label", "risk_score", "trend_slope"}
    required_levels = {"district", "state", "date", "season", "level_mbgl"}
    if not required_summary.issubset(summary.columns) or not required_levels.issubset(levels.columns):
        raise HTTPException(status_code=503, detail="Published artifacts have an unsupported schema.")
    return summary, levels


@app.get("/api/v1/health", tags=["service"])
def health() -> dict[str, Any]:
    summary, levels = _public_frames()
    try:
        manifest = _read_manifest()
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "districts": int(summary[["district", "state"]].drop_duplicates().shape[0]),
        "observations": int(len(levels)),
        "data_source": manifest.get("source_used"),
        "generated_at": manifest.get("generated_at"),
    }


@app.get("/api/v1/summary/risk", tags=["risk"])
def risk_summary() -> dict[str, Any]:
    summary, _ = _public_frames()
    counts = summary["risk_label"].value_counts()
    return {
        "total_districts": int(summary[["district", "state"]].drop_duplicates().shape[0]),
        "counts": {label: int(counts.get(label, 0)) for label in LABELS},
    }


@app.get("/api/v1/districts", tags=["districts"])
def list_districts(
    state: str | None = Query(default=None, max_length=100),
    risk: str | None = Query(default=None, max_length=20),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    if risk is not None and risk not in LABELS:
        raise HTTPException(status_code=422, detail=f"risk must be one of: {', '.join(LABELS)}")
    summary, _ = _public_frames()
    result = summary.copy()
    if state:
        result = result[result["state"].astype(str).str.casefold() == state.strip().casefold()]
    if risk:
        result = result[result["risk_label"] == risk]
    result = result.sort_values(
        ["risk_score", "trend_slope", "district"],
        ascending=[False, False, True],
        kind="stable",
    )
    total = len(result)
    fields = [
        "district", "state", "risk_label", "risk_score", "level_mbgl",
        "trend_slope", "predicted_decline_m_per_year",
    ]
    fields = [field for field in fields if field in result.columns]
    page = result.iloc[offset:offset + limit][fields]
    return {
        "items": _json_records(page),
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/v1/districts/{district}/series", tags=["districts"])
def district_series(
    district: str,
    state: str | None = Query(default=None, max_length=100),
) -> dict[str, Any]:
    if len(district) > 100:
        raise HTTPException(status_code=422, detail="district must be 100 characters or fewer.")
    _, levels = _public_frames()
    matches = levels[levels["district"].astype(str).str.casefold() == district.strip().casefold()]
    if state:
        matches = matches[matches["state"].astype(str).str.casefold() == state.strip().casefold()]
    if matches.empty:
        raise HTTPException(status_code=404, detail="District not found.")
    states = matches["state"].dropna().astype(str).unique()
    if len(states) > 1:
        raise HTTPException(
            status_code=409,
            detail="District name is ambiguous; specify the state query parameter.",
        )
    cols = ["district", "state", "date", "season", "level_mbgl"]
    if "n_stations" in matches.columns:
        cols.append("n_stations")
    matches = matches.sort_values("date")
    return {"items": _json_records(matches[cols])}
