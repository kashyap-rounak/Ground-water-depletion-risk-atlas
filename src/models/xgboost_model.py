"""XGBoost regression of next-season groundwater level, with walk-forward CV."""
from __future__ import annotations

import logging
from math import ceil
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from src.features.build_features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _entity_keys(df: pd.DataFrame) -> list[str]:
    """Use state-scoped district series when state identifiers are available."""
    return (
        ["state", "district"]
        if "state" in df.columns and df["state"].notna().any()
        else ["district"]
    )


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
    """Split by date and purge training rows whose targets enter the holdout."""
    if test_seasons < 1:
        raise ValueError("test_seasons must be at least 1.")
    _validate_supervised_frame(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = np.sort(df["date"].unique())
    if len(dates) <= test_seasons:
        test_dates = dates[-max(1, len(dates) // 5):]
    else:
        test_dates = dates[-int(test_seasons):]
    if not len(test_dates):
        raise ValueError("Chronological split requires at least one observation date.")

    test_start = pd.Timestamp(test_dates[0])
    target_dates = _next_observation_dates(df)
    test = df[df["date"].isin(test_dates)]
    train_mask = (df["date"] < test_start) & (target_dates < test_start)
    train = df.loc[train_mask]
    if train.empty or test.empty:
        raise ValueError("Not enough chronological observations for non-empty train and test sets.")
    return train, test


def _validate_supervised_frame(df: pd.DataFrame) -> None:
    required = {"district", "date", "target_next_level", *FEATURE_COLUMNS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing supervised-model columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Cannot train or evaluate on an empty dataset.")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Supervised dataset contains missing or invalid dates.")
    if df["district"].isna().any():
        raise ValueError("Supervised dataset contains rows without a district.")
    if df["target_next_level"].isna().any():
        raise ValueError("Supervised dataset contains rows without a next-level target.")


def _next_observation_dates(df: pd.DataFrame) -> pd.Series:
    """Return each district-season row's next observed date, aligned by index."""
    entity_keys = _entity_keys(df)
    ordered = df[[*entity_keys, "date"]].copy()
    ordered["_row_position"] = np.arange(len(ordered))
    ordered = ordered.sort_values([*entity_keys, "date"])
    next_dates = ordered.groupby(entity_keys, sort=False)["date"].shift(-1)

    aligned = pd.Series(index=np.arange(len(df)), dtype="datetime64[ns]")
    aligned.loc[ordered["_row_position"].to_numpy()] = pd.to_datetime(next_dates).to_numpy()
    aligned.index = df.index
    return aligned


def _walk_forward_splits(
    df: pd.DataFrame, n_splits: int
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield date-based panel folds with training targets purged at each cutoff."""
    dates = np.sort(df["date"].unique())
    if n_splits < 2 or len(dates) <= n_splits:
        raise ValueError(
            f"Walk-forward validation needs more than {n_splits} distinct dates; "
            f"received {len(dates)}."
        )

    target_dates = _next_observation_dates(df)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for _, validation_positions in splitter.split(dates):
        validation_dates = dates[validation_positions]
        validation_start = pd.Timestamp(validation_dates[0])
        training_mask = (df["date"] < validation_start) & (target_dates < validation_start)
        training = df.loc[training_mask]
        validation = df[df["date"].isin(validation_dates)]
        if training.empty or validation.empty:
            raise ValueError("A walk-forward fold has an empty training or validation set.")
        yield training, validation


def train_xgb_regressor(
    df: pd.DataFrame, cfg_xgb: dict
) -> tuple[XGBRegressor, dict[str, float], np.ndarray, pd.DataFrame]:
    """Select tree count on validation data, refit, and evaluate on a holdout.

    Returns (model, metrics, y_test_pred, test_frame). The test period is not
    used for early stopping or any other model selection.
    """
    from src.models.evaluate import regression_metrics

    train, test = chronological_split(df, int(cfg_xgb.get("test_seasons", 12)))
    training_dates = np.sort(train["date"].unique())
    validation_count = max(1, ceil(len(training_dates) * 0.2))
    validation_dates = training_dates[-validation_count:]
    validation_start = pd.Timestamp(validation_dates[0])
    next_dates = _next_observation_dates(train)
    fit_mask = (train["date"] < validation_start) & (next_dates < validation_start)
    fit = train.loc[fit_mask]
    validation = train[train["date"].isin(validation_dates)]
    if fit.empty or validation.empty:
        raise ValueError("Not enough dates to create a training and early-stopping validation set.")

    early_stop_model = _make_model(cfg_xgb, early_stopping=True)
    early_stop_model.fit(
        fit[FEATURE_COLUMNS],
        fit["target_next_level"],
        eval_set=[(validation[FEATURE_COLUMNS], validation["target_next_level"])],
        verbose=False,
    )
    best_iteration = getattr(early_stop_model, "best_iteration", None)
    selected_estimators = (
        int(best_iteration) + 1
        if best_iteration is not None
        else int(cfg_xgb.get("n_estimators", 600))
    )
    final_cfg = {**cfg_xgb, "n_estimators": selected_estimators}
    model = _make_model(final_cfg)
    X_train, y_train = train[FEATURE_COLUMNS], train["target_next_level"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target_next_level"]
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = regression_metrics(y_test, pred)
    baseline_mask = X_test["level_lag1"].notna()
    if baseline_mask.any():
        metrics["baseline_persistence_mae"] = float(
            np.mean(
                np.abs(
                    y_test.loc[baseline_mask]
                    - X_test.loc[baseline_mask, "level_lag1"]
                )
            )
        )
    else:
        metrics["baseline_persistence_mae"] = float("nan")
    logger.info("XGBoost holdout metrics: %s", metrics)
    return model, metrics, pred, test


def cross_validate(
    df: pd.DataFrame, cfg_xgb: dict, n_splits: int = 5
) -> dict[str, float]:
    """TimeSeriesSplit CV metrics across the training window."""
    from src.models.evaluate import regression_metrics

    train, _ = chronological_split(df, int(cfg_xgb.get("test_seasons", 12)))
    scores = {"mae": [], "rmse": [], "r2": []}
    for fold, (fold_train, validation) in enumerate(_walk_forward_splits(train, n_splits), 1):
        m = _make_model(cfg_xgb, early_stopping=False)
        m.fit(
            fold_train[FEATURE_COLUMNS],
            fold_train["target_next_level"],
        )
        p = m.predict(validation[FEATURE_COLUMNS])
        met = regression_metrics(validation["target_next_level"], p)
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
    entity_keys = _entity_keys(features)
    for entity_values, grp in features.groupby(entity_keys, sort=True, dropna=False):
        g = grp.sort_values("date").copy()
        last = g.iloc[-1]
        last_date = pd.Timestamp(last["date"])
        rolling_rain = float(last.get("rainfall_rolling3y", np.nan))
        entity_values = entity_values if isinstance(entity_values, tuple) else (entity_values,)
        entity = dict(zip(entity_keys, entity_values))
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
                    **entity, "date": future_date,
                    "season": seasons_cycle[(step - 1) % 2],
                    "forecast_level": yhat, "horizon": step,
                }
            )
    return pd.DataFrame(out_rows)
