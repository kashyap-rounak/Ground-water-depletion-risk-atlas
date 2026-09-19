"""End-to-end pipeline: ingest -> clean -> aggregate -> features ->
EDA -> SARIMA -> XGBoost -> risk -> SHAP -> artifacts.

Run:  python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.ingest import load_data
from src.data.clean import clean_data
from src.data.aggregate import aggregate_to_district
from src.features.build_features import build_features, drop_rows_without_target, FEATURE_COLUMNS
from src.eda import eda
from src.models.sarima import forecast_all_districts, backtest_sarima
from src.models import xgboost_model
from src.models.risk import assign_risk_labels, train_risk_classifier, LABELS
from src.models.explain import shap_values_classifier, global_importance
from src.geo.geo import load_boundaries, build_centroids, match_predictions, save_risk_geojson, normalize_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %s (%d rows)", path.name, len(df))


def main() -> None:
    cfg = load_config()
    art = Path(cfg.artifacts_dir)
    proc = Path(cfg.processed_dir)
    figs = Path(cfg.figures_dir)
    for d in (art, proc, figs):
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. ingest ------------------------------------------
    raw, source_used = load_data(cfg)
    logger.info("Data source: %s | rows=%d | districts=%d",
                source_used, len(raw), raw["district"].nunique())

    # ---------------- 2. clean -------------------------------------------
    clean = clean_data(raw, cfg)

    # ---------------- 3. aggregate ---------------------------------------
    levels, rainfall = aggregate_to_district(
        clean,
        pre_months=tuple(cfg.seasons.get("pre_monsoon_months", [3, 4, 5])),
        post_months=tuple(cfg.seasons.get("post_monsoon_months", [10, 11, 12])),
    )

    # ---------------- 4. features ----------------------------------------
    feats = build_features(
        levels, rainfall,
        rolling_years=int(cfg.features.get("rainfall_rolling_years", 3)),
        yoy_seasons=int(cfg.features.get("yoy_change_seasons", 2)),
    )

    # ---------------- 5. EDA ----------------------------------------------
    trend = eda.national_trend(levels)
    decline = eda.depletion_ranking(levels)
    corr = eda.rainfall_gw_correlation(levels, rainfall)
    eda.fig_national_trend(trend).write_html(figs / "national_trend.html")
    eda.fig_hotspots(decline).write_html(figs / "hotspots.html")
    logger.info("EDA: %d districts ranked by depletion", len(decline))

    # ---------------- 6. SARIMA -------------------------------------------
    sar_cfg = cfg.sarima
    forecast, aic = forecast_all_districts(
        levels,
        order=tuple(sar_cfg.get("order", [1, 1, 1])),
        seasonal_order=tuple(sar_cfg.get("seasonal_order", [0, 1, 1, 2])),
        forecast_years=int(sar_cfg.get("forecast_years", 3)),
        max_districts=int(sar_cfg.get("max_districts", 0)),
    )
    bt = backtest_sarima(
        levels,
        order=tuple(sar_cfg.get("order", [1, 1, 1])),
        seasonal_order=tuple(sar_cfg.get("seasonal_order", [0, 1, 1, 2])),
    )
    from src.models.evaluate import regression_metrics
    if not bt.empty:
        sarima_metrics = regression_metrics(bt["y_true"], bt["y_pred"])
    else:
        sarima_metrics = {"mae": np.nan, "rmse": np.nan, "r2": np.nan}
    logger.info("SARIMA backtest (sampled districts): %s", sarima_metrics)

    # ---------------- 7. XGBoost ------------------------------------------
    xgb_cfg = cfg.xgboost
    sup = drop_rows_without_target(feats)
    model, metrics, pred, test = xgboost_model.train_xgb_regressor(sup, xgb_cfg)
    cv = xgboost_model.cross_validate(sup, xgb_cfg)
    future = xgboost_model.predict_future_levels(
        model, feats, forecast_years=int(sar_cfg.get("forecast_years", 3))
    )
    metrics_full = {**metrics, **cv}
    logger.info("XGBoost holdout+CV: %s", metrics_full)

    # ---------------- 8. risk labels + classifier --------------------------
    latest = (
        feats.sort_values("date").groupby("district").tail(1)
        .set_index("district")
    )
    district_summary = pd.DataFrame({
        "district": decline["district"],
        "state": decline["state"],
        "trend_slope": decline["decline_m_per_year"],
        "level_mbgl": latest["level_mbgl"].reindex(decline["district"]).values,
        "annual_rainfall_mm": latest["annual_rainfall_mm"].reindex(decline["district"]).values,
        "rainfall_rolling3y": latest["rainfall_rolling3y"].reindex(decline["district"]).values,
        "season_flag": latest["season_flag"].reindex(decline["district"]).values,
    })
    # predicted decline from XGBoost future path: slope of forecast levels
    fc_slope = (
        future.sort_values("date")
        .groupby("district")
        .apply(
            lambda g: np.polyfit(
                np.arange(len(g)), g["forecast_level"].astype(float).values, 1
            )[0] * 2.0,  # per season -> per year
            include_groups=False,
        )
        .rename("predicted_decline_m_per_year")
        .reset_index()
    )
    district_summary = district_summary.merge(fc_slope, on="district", how="left")
    district_summary = assign_risk_labels(district_summary, cfg.risk.get("decline_m_per_year", {}))

    clf, report, cm, feature_cols = train_risk_classifier(district_summary, cfg.risk)

    # ---------------- 9. SHAP ------------------------------------------------
    X_all = district_summary[feature_cols].fillna(0)
    sv, _ = shap_values_classifier(clf, X_all)
    importance = global_importance(sv, feature_cols)
    logger.info("SHAP global importance:\n%s", importance)

    # ---------------- 10. persist artifacts -----------------------------------
    _save(clean, proc / "station_clean.parquet")
    _save(levels, proc / "district_levels.parquet")
    _save(rainfall, proc / "district_rainfall.parquet")
    _save(feats, proc / "features.parquet")
    _save(forecast, art / "sarima_forecast.parquet")
    _save(future, art / "xgb_future_levels.parquet")
    _save(district_summary, art / "district_summary.parquet")
    _save(decline, art / "depletion_ranking.parquet")
    _save(corr, art / "rainfall_gw_correlation.parquet")
    _save(pd.DataFrame([metrics_full]), art / "xgb_metrics.parquet")
    _save(pd.DataFrame([sarima_metrics]), art / "sarima_metrics.parquet")
    _save(importance, art / "shap_importance.parquet")

    with open(art / "xgb_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(art / "risk_classifier.pkl", "wb") as fh:
        pickle.dump(clf, fh)

    with open(art / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "xgboost": metrics_full,
                "sarima_backtest": sarima_metrics,
                "risk_classifier_report": report,
                "confusion_matrix": cm.tolist(),
                "risk_label_counts": district_summary["risk_label"].value_counts().to_dict(),
            },
            fh, indent=2, default=str,
        )
    logger.info("Metrics written to %s", art / "metrics.json")

    # ---------------- 11. geo -------------------------------------------------
    build_centroids(cfg, clean)
    gdf = load_boundaries(cfg)
    if gdf is not None:
        merged, unmatched, centroids = match_predictions(district_summary, gdf)
        path = save_risk_geojson(merged, figs)
        if path is not None:
            logger.info(
                "Choropleth geometry matched for %d districts (%d unmatched -> centroid fallback)",
                merged["district"].nunique(), len(unmatched),
            )
        if centroids is not None:
            centroids.to_parquet(proc / "district_centroids_geo.parquet", index=False)
            logger.info("Saved polygon centroids for %d districts", len(centroids))

    logger.info("Pipeline complete. Artifacts in %s", art)


if __name__ == "__main__":
    main()
