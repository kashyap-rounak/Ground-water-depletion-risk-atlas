"""SHAP explanations for the tree models (global + per-district)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def shap_values_classifier(clf, X: pd.DataFrame):
    """Return SHAP values (n, features, classes) for an XGBClassifier."""
    import shap

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap: list of (n, f) per class
        sv = np.stack(sv, axis=-1)  # -> (n, f, classes)
    return sv, explainer


def district_risk_explanation(
    clf,
    X_row: pd.DataFrame,
    class_names: list[str],
    max_display: int = 10,
) -> list[dict]:
    """Per-district SHAP explanation of the predicted class.

    Returns a list of {feature, shap_value, value} for the predicted class,
    sorted by |shap| descending.
    """
    import shap

    sv, _ = shap_values_classifier(clf, X_row)
    proba = clf.predict_proba(X_row)[0]
    pred_class = int(np.argmax(proba))
    vals = sv if isinstance(sv, np.ndarray) and sv.ndim == 2 else sv[0]
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        vals = sv[0, :, pred_class]
    contrib = (
        pd.DataFrame({"feature": X_row.columns, "shap_value": np.asarray(vals).ravel()})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(max_display)
    )
    contrib["value"] = [X_row.iloc[0][f] for f in contrib["feature"]]
    contrib["predicted_class"] = class_names[pred_class]
    contrib["class_probability"] = float(proba[pred_class])
    return contrib.to_dict("records")


def global_importance(sv: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature across samples (multi-class aware)."""
    arr = np.asarray(sv)
    if arr.ndim == 3:  # (n, features, classes)
        arr = np.abs(arr).mean(axis=(0, 2))
    else:
        arr = np.abs(arr).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": arr})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
