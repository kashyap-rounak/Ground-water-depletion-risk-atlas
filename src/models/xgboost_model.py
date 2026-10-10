"""XGBoost regression of next-season groundwater level, with walk-forward CV."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.features.build_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _make_model(cfg_xgb: dict, early_stopping: bool = False) -> XGBRegressor:
    params = dict(
        n_estimators=int(cfg_xgb.get("n_estimators", 600)),
        learning_rate=float(cfg_xgb.get("learning_rate", 0.03)),
        max_depth=int(cfg_xgb.get("max_depth", 5)),
        subsample=float(cfg_xgb.get("subsample", 0.85)),
        colsample_bytree=float(cfg_xgb.get("colsample_bytree", 0.85)),
        reg_lambda=float(cfg_xgb.get("reg_lambda", 1.0)),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
    )
    if early_stopping:
        params["early_stopping_rounds"] = int(cfg_xgb.get("early_stopping_rounds", 50))
    return XGBRegressor(**params)


def chronological_split(df: pd.DataFrame, test_seasons: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date: last `test_seasons` distinct dates are the holdout."""
    dates = np.sort(df["date"].unique())
    if len(dates) <= test_seasons:
        test_dates = dates[-max(1, len(dates) // 5):]
    else:
        test_dates = dates[-int(test_seasons):]
    test = df[df["date"].isin(test_dates)]
    train = df[~df["date"].isin(test_dates)]
    return train, test


def train_xgb_regressor(
    df: pd.DataFrame, cfg_xgb: dict
) -> tuple[XGBRegressor, dict[str, float], np.ndarray]:
    """Train on all rows with a valid target; evaluate on chronological holdout.

    Returns (model, metrics, y_test_pred, test_frame).
    """
    from src.models.evaluate import regression_metrics

    train, test = chronological_split(df, int(cfg_xgb.get("test_seasons", 12)))
    X_train, y_train = train[FEATURE_COLUMNS], train["target_next_level"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_next_level"]

    model = _make_model(cfg_xgb, early_stopping=True)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    metrics["baseline_persistence_mae"] = float(np.mean(np.abs(y_test - X_test["level_lag1"])))
    logger.info("XGBoost holdout metrics: %s", metrics)
    return model, metrics, pred, test


def cross_validate(
    df: pd.DataFrame, cfg_xgb: dict, n_splits: int = 5
) -> dict[str, float]:
    """TimeSeriesSplit CV metrics across the training window."""
    from src.models.evaluate import regression_metrics

    train, _ = chronological_split(df, int(cfg_xgb.get("test_seasons", 12)))
    X, y = train[FEATURE_COLUMNS], train["target_next_level"]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = {"mae": [], "rmse": [], "r2": []}
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        m = _make_model(cfg_xgb, early_stopping=False)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[va])
        met = regression_metrics(y.iloc[va], p)
        for k in scores:
            scores[k].append(met[k])
        logger.debug("Fold %d: %s", fold, met)
    out = {f"cv_{k}_mean": float(np.mean(v)) for k, v in scores.items()}
    logger.info("XGBoost CV: %s", out)
    return out


def predict_future_levels(
    model: XGBRegressor,
    features: pd.DataFrame,
    forecast_years: int = 3,
) -> pd.DataFrame:
    """Recursive multi-step prediction of future district-season levels.

    Uses SARIMA-style future dates; rainfall for future seasons is assumed
    equal to the district's trailing 3-year mean (documented assumption).
    """
    seasons_cycle = ["pre-monsoon", "post-monsoon"]
    out_rows = []
    for district, grp in features.groupby("district"):
        g = grp.sort_values("date").copy()
        last = g.iloc[-1]
        last_date = pd.Timestamp(last["date"])
        rolling_rain = float(last.get("rainfall_rolling3y", np.nan))
        state = last["state"]
        lag1, lag2, lag4 = float(last["level_mbgl"]), float(last["level_lag1"]), float(last["level_lag4"])
        series = list(g["level_mbgl"].astype(float).values)

        for step in range(1, forecast_years * 2 + 1):
            month = 5 if seasons_cycle[(step - 1) % 2] == "pre-monsoon" else 12
            year = last_date.year + (step - 1) // 2 + (1 if month < last_date.month else 0)
            future_date = pd.Timestamp(year=year, month=month, day=1)
            season_flag = int(seasons_cycle[(step - 1) % 2] == "post-monsoon")
            row = {
                "level_lag1": series[-1],
                "level_lag2": series[-2] if len(series) >= 2 else series[-1],
                "level_lag4": series[-4] if len(series) >= 4 else series[-1],
                "level_change_yoy": series[-1] - (series[-2] if len(series) >= 2 else series[-1]),
                "rainfall_rolling3y": rolling_rain,
                "rainfall_lag1_season": rolling_rain / 2.0,
                "rainfall_lag1_year": rolling_rain,
                "season_flag": season_flag,
                "season_sin": np.sin(2 * np.pi * season_flag),
                "season_cos": np.cos(2 * np.pi * season_flag),
                "trend_slope": float(last["trend_slope"]) if pd.notna(last["trend_slope"]) else 0.0,
            }
            X = pd.DataFrame([row])[FEATURE_COLUMNS]
            yhat = float(model.predict(X)[0])
            series.append(yhat)
            out_rows.append(
                {
                    "district": district, "state": state, "date": future_date,
                    "season": seasons_cycle[(step - 1) % 2],
                    "forecast_level": yhat, "horizon": step,
                }
            )
    return pd.DataFrame(out_rows)
