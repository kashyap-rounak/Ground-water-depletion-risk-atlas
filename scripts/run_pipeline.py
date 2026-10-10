"""End-to-end pipeline: ingest -> clean -> aggregate -> features ->
EDA -> SARIMA -> XGBoost -> risk -> SHAP -> artifacts.

Run:  python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config
from src.data.ingest import load_data
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district
from src.features.build_features import build_features, drop_rows_without_target
from src.eda import eda
from src.models.sarima import forecast_all_districts, backtest_sarima
from src.models import xgboost_model
from src.models.evaluate import regression_metrics
from src.models.risk import assign_risk_labels, train_risk_classifier
from src.models.explain import shap_values_classifier, global_importance
from src.geo.geo import (
    load_boundaries,
    build_centroids,
    combine_centroids,
    match_predictions,
    save_risk_geojson,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


@dataclass
class PreparedData:
    """Data products shared by the modeling and reporting stages."""

    clean: pd.DataFrame
    levels: pd.DataFrame
    rainfall: pd.DataFrame
    features: pd.DataFrame
    decline: pd.DataFrame
    correlation: pd.DataFrame
    source_used: str


@dataclass
class ForecastResults:
    """Forecasts, trained regressor, and regression evaluation outputs."""

    sarima: pd.DataFrame
    xgb_future: pd.DataFrame
    xgb_model: XGBRegressor
    xgb_metrics: dict[str, float]
    sarima_metrics: dict[str, float]


@dataclass
class RiskResults:
    """District risk output and its classifier/explanation artifacts."""

    district_summary: pd.DataFrame
    classifier: XGBClassifier
    report: str
    confusion_matrix: np.ndarray
    feature_columns: list[str]
    shap_importance: pd.DataFrame


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s (%d rows)", path.name, len(df))


def _prepare_data(cfg: Config, figures_dir: Path) -> PreparedData:
    """Load, clean, aggregate, engineer features, and calculate EDA outputs."""
    raw, source_used = load_data(cfg)
    logger.info(
        "Data source: %s | rows=%d | districts=%d",
        source_used,
        len(raw),
        raw["district"].nunique(),
    )
    clean = clean_data(raw, cfg)
    levels, rainfall = aggregate_to_district(
        clean,
        pre_months=tuple(cfg.seasons.get("pre_monsoon_months", [3, 4, 5])),
        post_months=tuple(cfg.seasons.get("post_monsoon_months", [10, 11, 12])),
    )
    features = build_features(
        levels,
        rainfall,
        rolling_years=int(cfg.features.get("rainfall_rolling_years", 3)),
        yoy_seasons=int(cfg.features.get("yoy_change_seasons", 2)),
    )

    trend = eda.national_trend(levels)
    decline = eda.depletion_ranking(levels)
    correlation = eda.rainfall_gw_correlation(levels, rainfall)
    eda.fig_national_trend(trend).write_html(figures_dir / "national_trend.html")
    eda.fig_hotspots(decline).write_html(figures_dir / "hotspots.html")
    logger.info("EDA: %d districts ranked by depletion", len(decline))

    return PreparedData(
        clean=clean,
        levels=levels,
        rainfall=rainfall,
        features=features,
        decline=decline,
        correlation=correlation,
        source_used=source_used,
    )


def _run_forecasts(prepared: PreparedData, cfg: Config) -> ForecastResults:
    """Fit the configured forecasters and calculate their evaluation metrics."""
    sar_cfg = cfg.sarima
    order = tuple(sar_cfg.get("order", [1, 1, 1]))
    seasonal_order = tuple(sar_cfg.get("seasonal_order", [0, 1, 1, 2]))
    sarima, _ = forecast_all_districts(
        prepared.levels,
        order=order,
        seasonal_order=seasonal_order,
        forecast_years=int(sar_cfg.get("forecast_years", 3)),
        max_districts=int(sar_cfg.get("max_districts", 0)),
    )
    backtest = backtest_sarima(
        prepared.levels,
        order=order,
        seasonal_order=seasonal_order,
    )
    if not backtest.empty:
        sarima_metrics = regression_metrics(backtest["y_true"], backtest["y_pred"])
    else:
        sarima_metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    logger.info("SARIMA backtest (sampled districts): %s", sarima_metrics)

    supervised = drop_rows_without_target(prepared.features)
    model, metrics, _, _ = xgboost_model.train_xgb_regressor(supervised, cfg.xgboost)
    cv_metrics = xgboost_model.cross_validate(supervised, cfg.xgboost)
    future = xgboost_model.predict_future_levels(
        model,
        prepared.features,
        forecast_years=int(sar_cfg.get("forecast_years", 3)),
    )
    xgb_metrics = {**metrics, **cv_metrics}
    logger.info("XGBoost holdout+CV: %s", xgb_metrics)

    return ForecastResults(
        sarima=sarima,
        xgb_future=future,
        xgb_model=model,
        xgb_metrics=xgb_metrics,
        sarima_metrics=sarima_metrics,
    )


def _build_district_summary(
    features: pd.DataFrame,
    decline: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Combine latest observed features and forecast trends per district."""
    entity_keys = (
        ["state", "district"]
        if "state" in features.columns and features["state"].notna().any()
        else ["district"]
    )
    latest = (
        features.sort_values("date")
        .groupby(entity_keys, dropna=False)
        .tail(1)
    )
    district_summary = decline.rename(
        columns={"decline_m_per_year": "trend_slope"}
    ).merge(
        latest[
            [
                *entity_keys,
                "level_mbgl",
                "annual_rainfall_mm",
                "rainfall_rolling3y",
                "season_flag",
            ]
        ],
        on=entity_keys,
        how="left",
    )

    forecast_keys = (
        ["state", "district"]
        if "state" in future.columns and future["state"].notna().any()
        else ["district"]
    )
    forecast_slope = (
        future.sort_values("date")
        .groupby(forecast_keys, dropna=False)
        .apply(
            lambda group: np.polyfit(
                np.arange(len(group)), group["forecast_level"].astype(float).values, 1
            )[0]
            * 2.0,
            include_groups=False,
        )
        .rename("predicted_decline_m_per_year")
        .reset_index()
    )
    return district_summary.merge(forecast_slope, on=forecast_keys, how="left")


def _run_risk_analysis(
    prepared: PreparedData,
    forecasts: ForecastResults,
    cfg: Config,
) -> RiskResults:
    """Assign risk labels, train the classifier, and calculate SHAP importance."""
    district_summary = _build_district_summary(
        prepared.features,
        prepared.decline,
        forecasts.xgb_future,
    )
    district_summary = assign_risk_labels(
        district_summary,
        cfg.risk.get("decline_m_per_year", {}),
    )
    classifier, report, confusion_matrix, feature_columns = train_risk_classifier(
        district_summary,
        cfg.risk,
    )

    explain_data = district_summary[feature_columns].fillna(0)
    shap_values, _ = shap_values_classifier(classifier, explain_data)
    importance = global_importance(shap_values, feature_columns)
    logger.info("SHAP global importance:\n%s", importance)

    return RiskResults(
        district_summary=district_summary,
        classifier=classifier,
        report=report,
        confusion_matrix=confusion_matrix,
        feature_columns=feature_columns,
        shap_importance=importance,
    )


def _save_artifacts(
    prepared: PreparedData,
    forecasts: ForecastResults,
    risks: RiskResults,
    cfg: Config,
    processed_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Persist pipeline tables, trained models, and dashboard metrics."""
    _save(prepared.clean, processed_dir / "station_clean.parquet")
    _save(prepared.levels, processed_dir / "district_levels.parquet")
    _save(prepared.rainfall, processed_dir / "district_rainfall.parquet")
    _save(prepared.features, processed_dir / "features.parquet")
    _save(forecasts.sarima, artifacts_dir / "sarima_forecast.parquet")
    _save(forecasts.xgb_future, artifacts_dir / "xgb_future_levels.parquet")
    _save(risks.district_summary, artifacts_dir / "district_summary.parquet")
    _save(prepared.decline, artifacts_dir / "depletion_ranking.parquet")
    _save(prepared.correlation, artifacts_dir / "rainfall_gw_correlation.parquet")
    _save(pd.DataFrame([forecasts.xgb_metrics]), artifacts_dir / "xgb_metrics.parquet")
    _save(pd.DataFrame([forecasts.sarima_metrics]), artifacts_dir / "sarima_metrics.parquet")
    _save(risks.shap_importance, artifacts_dir / "shap_importance.parquet")

    with open(artifacts_dir / "xgb_model.pkl", "wb") as fh:
        pickle.dump(forecasts.xgb_model, fh)
    with open(artifacts_dir / "risk_classifier.pkl", "wb") as fh:
        pickle.dump(risks.classifier, fh)

    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "xgboost": forecasts.xgb_metrics,
                "sarima_backtest": forecasts.sarima_metrics,
                "risk_classifier_report": risks.report,
                "confusion_matrix": risks.confusion_matrix.tolist(),
                "risk_label_counts": (
                    risks.district_summary["risk_label"].value_counts().to_dict()
                ),
            },
            fh,
            indent=2,
            default=str,
        )
    logger.info("Metrics written to %s", artifacts_dir / "metrics.json")


def _save_run_metadata(
    prepared: PreparedData,
    cfg: Config,
    artifacts_dir: Path,
) -> None:
    """Record data provenance only after the pipeline has completed."""
    groundwater = prepared.clean.loc[
        prepared.clean["level_mbgl"].notna()
    ]
    date_min = groundwater["date"].min()
    date_max = groundwater["date"].max()
    groundwater_entities = groundwater[["state", "district"]].drop_duplicates()
    configured_months = set(cfg.seasons.get("pre_monsoon_months", [3, 4, 5]))
    configured_months.update(cfg.seasons.get("post_monsoon_months", [10, 11, 12]))
    groundwater_months = groundwater["date"].dt.month
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_used": prepared.source_used,
        "configured_source": cfg.data_source,
        "rows": int(len(prepared.clean)),
        "groundwater_observations": int(len(groundwater)),
        "districts": int(len(groundwater_entities)),
        "seasonal_groundwater_observations": int(
            groundwater_months.isin(configured_months).sum()
        ),
        "groundwater_observations_excluded_by_season": int(
            (~groundwater_months.isin(configured_months)).sum()
        ),
        "date_min": date_min.isoformat() if pd.notna(date_min) else None,
        "date_max": date_max.isoformat() if pd.notna(date_max) else None,
    }
    if prepared.source_used == "local":
        provenance = cfg.data_cfg.get("provenance")
        if provenance:
            run_metadata["provenance"] = provenance
            source_ranges = provenance.get("source_date_ranges")
            if source_ranges:
                run_metadata["source_date_ranges"] = source_ranges
    with open(artifacts_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, indent=2)


def _build_geospatial_artifacts(
    cfg: Config,
    prepared: PreparedData,
    risks: RiskResults,
    processed_dir: Path,
    figures_dir: Path,
) -> None:
    """Build centroid and matched-boundary outputs for the dashboard."""
    station_centroids = build_centroids(cfg, prepared.clean)
    logger.info("Saved station centroids for %d districts", len(station_centroids))
    boundaries = load_boundaries(cfg)
    if boundaries is None:
        station_centroids.to_parquet(
            processed_dir / "district_centroids_geo.parquet",
            index=False,
        )
        return

    merged, unmatched, centroids = match_predictions(risks.district_summary, boundaries)
    path = save_risk_geojson(merged, figures_dir)
    if path is not None:
        logger.info(
            "Choropleth geometry matched for %d districts (%d unmatched -> centroid fallback)",
            merged["district"].nunique(),
            len(unmatched),
        )
    combined_centroids = combine_centroids(centroids, station_centroids)
    combined_centroids.to_parquet(
        processed_dir / "district_centroids_geo.parquet",
        index=False,
    )
    logger.info(
        "Saved %d centroid locations (%d from matched boundaries)",
        len(combined_centroids),
        0 if centroids is None else len(centroids),
    )


def main(cfg: Config | None = None) -> None:
    """Run the configured data, modeling, and artifact pipeline."""
    cfg = cfg or load_config()
    artifacts_dir = Path(cfg.artifacts_dir)
    processed_dir = Path(cfg.processed_dir)
    figures_dir = Path(cfg.figures_dir)
    for directory in (artifacts_dir, processed_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_data(cfg, figures_dir)
    forecasts = _run_forecasts(prepared, cfg)
    risks = _run_risk_analysis(prepared, forecasts, cfg)
    _save_artifacts(prepared, forecasts, risks, cfg, processed_dir, artifacts_dir)
    _build_geospatial_artifacts(cfg, prepared, risks, processed_dir, figures_dir)
    _save_run_metadata(prepared, cfg, artifacts_dir)
    logger.info("Pipeline complete. Artifacts in %s", artifacts_dir)


if __name__ == "__main__":
    main()
