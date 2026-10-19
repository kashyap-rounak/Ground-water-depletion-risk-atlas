"""FastAPI service for the Groundwater Depletion Risk Atlas.

Run locally:
    uvicorn src.api.main:app --reload --port 8000
Interactive OpenAPI documentation available at:
    http://localhost:8000/docs
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.geo.geo import normalize_name

app = FastAPI(
    title="🌍 Groundwater Depletion Risk Atlas API",
    description="REST API for querying historical groundwater levels, SARIMA/XGBoost forecasts, and 4-tier depletion risk classifications across India.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend or cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ Data Loaders
@lru_cache(maxsize=1)
def get_data() -> Dict[str, Any]:
    """Load and cache precomputed pipeline artifacts."""
    cfg = load_config()
    proc = Path(cfg.processed_dir)
    art = Path(cfg.artifacts_dir)

    def _read_pq(p: Path) -> Optional[pd.DataFrame]:
        return pd.read_parquet(p) if p.exists() else None

    metrics_file = art / "metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}

    return {
        "cfg": cfg,
        "levels": _read_pq(proc / "district_levels.parquet"),
        "rainfall": _read_pq(proc / "district_rainfall.parquet"),
        "sarima": _read_pq(art / "sarima_forecast.parquet"),
        "future": _read_pq(art / "xgb_future_levels.parquet"),
        "summary": _read_pq(art / "district_summary.parquet"),
        "decline": _read_pq(art / "depletion_ranking.parquet"),
        "shap": _read_pq(art / "shap_importance.parquet"),
        "metrics": metrics,
    }


def _clean_record(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert NaNs and NumPy types to JSON-safe Python types."""
    clean = {}
    for k, v in d.items():
        if pd.isna(v) or v is None:
            clean[k] = None
        elif isinstance(v, (np.floating, float)):
            clean[k] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.strftime("%Y-%m-%d")
        else:
            clean[k] = v
    return clean


def _find_district_row(df: pd.DataFrame, district_name: str) -> pd.DataFrame:
    """Case-insensitive and normalized matching for district name."""
    norm = normalize_name(district_name)
    matched = df[df["district"].map(normalize_name) == norm]
    if matched.empty:
        # direct case-insensitive fallback
        matched = df[df["district"].str.lower() == district_name.strip().lower()]
    return matched


# ------------------------------------------------------------------ Pydantic Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    artifacts_loaded: bool
    total_districts: int
    models_available: List[str]


class DistrictSummary(BaseModel):
    district: str
    state: Optional[str] = None
    risk_label: Optional[str] = None
    risk_score: Optional[int] = None
    level_mbgl: Optional[float] = None
    trend_slope: Optional[float] = None
    predicted_decline_m_per_year: Optional[float] = None
    annual_rainfall_mm: Optional[float] = None
    rainfall_rolling3y: Optional[float] = None


class ForecastItem(BaseModel):
    district: str
    date: str
    season: str
    forecast_level: float
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None
    model: str


class HistoryItem(BaseModel):
    district: str
    date: str
    year: int
    season: str
    level_mbgl: Optional[float] = None
    rainfall_mm: Optional[float] = None


class ShapFeature(BaseModel):
    feature: str
    mean_abs_shap: float


# ------------------------------------------------------------------ Endpoints
@app.get("/", tags=["General"])
def root() -> Dict[str, str]:
    """Root metadata and documentation links."""
    return {
        "title": "Groundwater Depletion Risk Atlas API",
        "documentation": "/docs",
        "health": "/api/v1/health",
        "districts": "/api/v1/districts",
        "hotspots": "/api/v1/hotspots",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    """Check API status and artifact availability."""
    data = get_data()
    summary = data["summary"]
    loaded = summary is not None and not summary.empty
    count = int(summary["district"].nunique()) if loaded else 0

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        version="1.0.0",
        artifacts_loaded=loaded,
        total_districts=count,
        models_available=["SARIMA", "XGBoost", "Risk_Classifier", "SHAP"],
    )


@app.get("/api/v1/districts", response_model=List[DistrictSummary], tags=["Districts"])
def list_districts(
    state: Optional[str] = Query(None, description="Filter by state name"),
    risk_label: Optional[str] = Query(None, description="Filter by risk tier: Low, Moderate, High, Very High"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
) -> List[DistrictSummary]:
    """Retrieve summary table of monitored districts with risk scores and current levels."""
    data = get_data()
    summary = data["summary"]
    if summary is None:
        raise HTTPException(status_code=503, detail="District summary data not generated yet.")

    df = summary.copy()
    if state:
        df = df[df["state"].str.lower() == state.strip().lower()]
    if risk_label:
        df = df[df["risk_label"].str.lower() == risk_label.strip().lower()]

    records = [_clean_record(r) for r in df.head(limit).to_dict(orient="records")]
    return [DistrictSummary(**r) for r in records]


@app.get("/api/v1/districts/{district}", response_model=DistrictSummary, tags=["Districts"])
def get_district_detail(district: str) -> DistrictSummary:
    """Retrieve full profile and risk assessment for a specific district."""
    data = get_data()
    summary = data["summary"]
    if summary is None:
        raise HTTPException(status_code=503, detail="Summary artifacts not loaded.")

    matched = _find_district_row(summary, district)
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found.")

    record = _clean_record(matched.iloc[0].to_dict())
    return DistrictSummary(**record)


@app.get("/api/v1/districts/{district}/forecast", response_model=List[ForecastItem], tags=["Forecasting"])
def get_district_forecast(
    district: str,
    model: str = Query("all", description="Model filter: 'sarima', 'xgboost', or 'all'"),
) -> List[ForecastItem]:
    """Retrieve 3-year forward seasonal water table predictions for a district."""
    data = get_data()
    items: List[ForecastItem] = []

    # 1. SARIMA forecasts
    if model.lower() in ("all", "sarima"):
        sarima_df = data["sarima"]
        if sarima_df is not None:
            matched_sarima = _find_district_row(sarima_df, district)
            for _, r in matched_sarima.iterrows():
                items.append(
                    ForecastItem(
                        district=r["district"],
                        date=pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                        season=r["season"],
                        forecast_level=round(float(r["forecast_level"]), 3),
                        lower_95=round(float(r["lower_95"]), 3) if "lower_95" in r and not pd.isna(r["lower_95"]) else None,
                        upper_95=round(float(r["upper_95"]), 3) if "upper_95" in r and not pd.isna(r["upper_95"]) else None,
                        model="SARIMA",
                    )
                )

    # 2. XGBoost future levels
    if model.lower() in ("all", "xgboost"):
        xgb_df = data["future"]
        if xgb_df is not None:
            matched_xgb = _find_district_row(xgb_df, district)
            for _, r in matched_xgb.iterrows():
                items.append(
                    ForecastItem(
                        district=r["district"],
                        date=pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                        season=r["season"],
                        forecast_level=round(float(r["forecast_level"]), 3),
                        lower_95=None,
                        upper_95=None,
                        model="XGBoost",
                    )
                )

    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for district '{district}' with model '{model}'.",
        )

    return items


@app.get("/api/v1/districts/{district}/history", response_model=List[HistoryItem], tags=["Historical Data"])
def get_district_history(district: str) -> List[HistoryItem]:
    """Retrieve historical seasonal groundwater levels and precipitation records."""
    data = get_data()
    levels_df = data["levels"]
    rainfall_df = data["rainfall"]

    if levels_df is None:
        raise HTTPException(status_code=503, detail="Historical levels data not loaded.")

    matched_levels = _find_district_row(levels_df, district)
    if matched_levels.empty:
        raise HTTPException(status_code=404, detail=f"District '{district}' not found in historical records.")

    m = matched_levels.copy()
    if rainfall_df is not None:
        matched_rain = _find_district_row(rainfall_df, district)
        if not matched_rain.empty:
            m = m.merge(
                matched_rain[["district", "year", "season", "rainfall_mm"]],
                on=["district", "year", "season"],
                how="left",
            )
        else:
            m["rainfall_mm"] = None
    else:
        m["rainfall_mm"] = None

    history: List[HistoryItem] = []
    for _, r in m.sort_values("date").iterrows():
        history.append(
            HistoryItem(
                district=r["district"],
                date=pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                year=int(r["year"]),
                season=r["season"],
                level_mbgl=round(float(r["level_mbgl"]), 3) if not pd.isna(r["level_mbgl"]) else None,
                rainfall_mm=round(float(r["rainfall_mm"]), 2) if not pd.isna(r.get("rainfall_mm")) else None,
            )
        )
    return history


@app.get("/api/v1/hotspots", response_model=List[DistrictSummary], tags=["Analysis"])
def get_depletion_hotspots(
    limit: int = Query(10, ge=1, le=100, description="Number of top depleting districts to return")
) -> List[DistrictSummary]:
    """Retrieve top priority depletion hotspots ranked by historical/projected drawdown rate."""
    data = get_data()
    decline_df = data["decline"]
    summary_df = data["summary"]

    if decline_df is None or summary_df is None:
        raise HTTPException(status_code=503, detail="Depletion rankings not loaded.")

    top_districts = decline_df.head(limit)["district"].tolist()
    top_summary = summary_df[summary_df["district"].isin(top_districts)].copy()
    # Preserve ranking order
    top_summary["rank"] = top_summary["district"].map(lambda d: top_districts.index(d) if d in top_districts else 999)
    top_summary = top_summary.sort_values("rank")

    records = [_clean_record(r) for r in top_summary.to_dict(orient="records")]
    return [DistrictSummary(**r) for r in records]


@app.get("/api/v1/explain/shap", response_model=List[ShapFeature], tags=["Explainability"])
def get_global_shap_importance() -> List[ShapFeature]:
    """Retrieve global SHAP feature importance driving the 4-tier risk classification engine."""
    data = get_data()
    shap_df = data["shap"]
    if shap_df is None:
        raise HTTPException(status_code=503, detail="SHAP importance artifact not loaded.")

    results: List[ShapFeature] = []
    for _, r in shap_df.iterrows():
        results.append(
            ShapFeature(
                feature=str(r["feature"]),
                mean_abs_shap=round(float(r["mean_abs_shap"]), 4),
            )
        )
    return results


@app.get("/api/v1/metrics", tags=["Analysis"])
def get_model_metrics() -> Dict[str, Any]:
    """Retrieve backtest, holdout, and walk-forward cross-validation performance metrics."""
    data = get_data()
    metrics = data["metrics"]
    if not metrics:
        raise HTTPException(status_code=503, detail="Model metrics file not found.")
    return metrics
