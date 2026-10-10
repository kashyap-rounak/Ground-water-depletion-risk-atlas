"""SARIMA time-series forecasting of groundwater levels per district.

The bi-annual series (pre/post monsoon) is modelled on an integer index
with seasonal period 2, avoiding irregular date-frequency issues; dates
are reconstructed for the forecast horizon.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fit_forecast_one(series: pd.Series, order, seasonal_order, steps: int):
    """Fit SARIMAX on a 1D series (integer index) and forecast `steps`."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            np.asarray(series, dtype=float),
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=steps)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
        return mean, ci, res


def _future_dates(last_date: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    """Next `steps` bi-annual season-end dates after `last_date`."""
    out: list[pd.Timestamp] = []
    year, month = last_date.year, last_date.month
    for _ in range(steps):
        if month == 5:  # last was pre-monsoon -> next is post-monsoon same year
            month = 12
        else:           # last was post-monsoon -> next is pre-monsoon next year
            year += 1
            month = 5
        out.append(pd.Timestamp(year=year, month=month, day=1))
    return out


def forecast_all_districts(
    levels: pd.DataFrame,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 2),
    forecast_years: int = 3,
    max_districts: int = 0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Forecast future levels for every district.

    Returns (forecast_df, fit_aic_by_district).
    forecast_df columns: district, date, season, forecast_level,
    lower_95, upper_95, horizon.
    """
    seasonal_period = int(seasonal_order[3])
    steps = int(forecast_years) * seasonal_period

    parts: list[pd.DataFrame] = []
    aic: dict[str, float] = {}
    districts = sorted(levels["district"].unique())
    if max_districts:
        districts = districts[: int(max_districts)]

    for district in districts:
        g = levels[levels["district"] == district].sort_values("date")
        s = g.set_index("date")["level_mbgl"].dropna()
        if len(s) < 4 * seasonal_period:
            logger.debug("Skipping %s: too few observations (%d)", district, len(s))
            continue
        try:
            mean, ci, res = _fit_forecast_one(s, order, seasonal_order, steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SARIMA failed for %s: %s", district, exc)
            continue
        aic[district] = float(res.aic)
        future_dates = _future_dates(pd.Timestamp(s.index[-1]), steps)
        parts.append(
            pd.DataFrame(
                {
                    "district": district,
                    "date": future_dates,
                    "forecast_level": mean,
                    "lower_95": ci[:, 0],
                    "upper_95": ci[:, 1],
                    "horizon": np.arange(1, steps + 1),
                }
            )
        )

    if not parts:
        raise RuntimeError("SARIMA produced no forecasts (all districts skipped).")
    fc = pd.concat(parts, ignore_index=True)
    fc["season"] = np.where(fc["date"].dt.month == 5, "pre-monsoon", "post-monsoon")
    fc["year"] = fc["date"].dt.year
    logger.info("SARIMA: forecasts for %d districts", fc["district"].nunique())
    return fc, aic


def backtest_sarima(
    levels: pd.DataFrame,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 2),
    holdout_seasons: int = 6,
    max_districts: int = 12,
) -> pd.DataFrame:
    """Rolling-origin backtest on a sample of districts for evaluation."""
    seasonal_period = int(seasonal_order[3])
    rows = []
    districts = sorted(levels["district"].unique())[: int(max_districts)]
    for district in districts:
        g = levels[levels["district"] == district].sort_values("date")
        s = g.set_index("date")["level_mbgl"].dropna()
        if len(s) < 4 * seasonal_period + holdout_seasons:
            continue
        train, test = s.iloc[:-holdout_seasons], s.iloc[-holdout_seasons:]
        try:
            mean, _, _ = _fit_forecast_one(train, order, seasonal_order, holdout_seasons)
        except Exception:  # noqa: BLE001
            continue
        for date, yhat, y in zip(test.index, mean, test.values):
            rows.append(
                {"district": district, "date": date, "y_true": float(y), "y_pred": float(yhat)}
            )
    return pd.DataFrame(rows)
