"""Risk classification: label construction + gradient-boosted classifier.

Risk labels combine two documented signals of groundwater depletion:
  - predicted next-year decline (m/yr) from the XGBoost level model
  - historical trend slope (m/yr) over the observed series

Thresholds (m/yr) come from config: risk.decline_m_per_year.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

LABELS = ["Low", "Moderate", "High", "Very High"]
LABEL_TO_INT = {lab: i for i, lab in enumerate(LABELS)}


def assign_risk_labels(
    districts: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """Add `risk_label` / `risk_score` (0-3) to a district summary table.

    Expects columns: district, predicted_decline_m_per_year, trend_slope.
    """
    d = districts.copy()
    th = thresholds or {}
    mod = float(th.get("moderate", 0.10))
    high = float(th.get("high", 0.30))
    vhigh = float(th.get("very_high", 0.60))

    # worst of predicted future decline and historical trend
    d["decline_signal"] = d[["predicted_decline_m_per_year", "trend_slope"]].max(axis=1)
    d["risk_score"] = pd.cut(
        d["decline_signal"],
        bins=[-np.inf, mod, high, vhigh, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)
    d["risk_label"] = d["risk_score"].map(dict(enumerate(LABELS)))
    return d


def train_risk_classifier(
    districts: pd.DataFrame, cfg_risk: dict, seed: int = 42
):
    """Fit XGBClassifier on district-level risk labels.

    Returns (model, report_text, confusion, feature_names, class_order).
    The model predicts risk_score (0..3) from hydrological features.
    """
    feature_cols = [
        "level_mbgl", "predicted_decline_m_per_year", "trend_slope",
        "annual_rainfall_mm", "rainfall_rolling3y", "season_flag",
    ]
    d = districts.dropna(subset=["risk_score"]).copy()
    X, y = d[feature_cols], d["risk_score"].astype(int)

    strat = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=strat
    )
    clf = XGBClassifier(
        n_estimators=int(cfg_risk.get("classifier", {}).get("n_estimators", 350)),
        learning_rate=float(cfg_risk.get("classifier", {}).get("learning_rate", 0.06)),
        max_depth=int(cfg_risk.get("classifier", {}).get("max_depth", 4)),
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
        objective="multi:softprob",
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    present = sorted(y.unique().tolist())
    report = classification_report(
        y_te, y_pred,
        labels=present,
        target_names=[LABELS[i] for i in present],
        zero_division=0,
    )
    cm = confusion_matrix(y_te, y_pred)
    logger.info("Risk classifier report:\n%s", report)
    return clf, report, cm, feature_cols
