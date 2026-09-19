"""Exploratory analysis: trends, seasonality, rainfall-groundwater links.

Every function returns plain pandas objects / plotly figures so the same
code drives both the pipeline's saved reports and the Streamlit dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data.aggregate import SEASON_ORDER


def national_trend(levels: pd.DataFrame) -> pd.DataFrame:
    """National mean groundwater depth per season (all districts averaged)."""
    return (
        levels.groupby(["year", "season"], as_index=False)["level_mbgl"]
        .mean()
        .sort_values(["year", "season"])
    )


def depletion_ranking(levels: pd.DataFrame) -> pd.DataFrame:
    """Districts ranked by per-year decline estimated over the full series."""
    rows = []
    for district, grp in levels.sort_values("date").groupby("district"):
        s = grp.dropna(subset=["level_mbgl"])
        if len(s) < 4:
            continue
        x = s["date"].map(pd.Timestamp.toordinal).astype(float)
        y = s["level_mbgl"].astype(float)
        slope_m_per_year = np.polyfit(x, y, 1)[0] * 365.25
        rows.append(
            {
                "district": district,
                "state": s["state"].iloc[0],
                "level_start": y.iloc[0],
                "level_end": y.iloc[-1],
                "decline_m_per_year": slope_m_per_year,
                "n_seasons": len(s),
            }
        )
    return pd.DataFrame(rows).sort_values("decline_m_per_year", ascending=False)


def seasonality(levels: pd.DataFrame) -> pd.DataFrame:
    """Mean level by season across all districts."""
    return levels.groupby("season")["level_mbgl"].agg(["mean", "std", "count"])


def rainfall_gw_correlation(levels: pd.DataFrame, rainfall: pd.DataFrame) -> pd.DataFrame:
    """Per-district correlation of annual rainfall with groundwater level."""
    ann = rainfall.groupby(["district", "year"], as_index=False)["annual_rainfall_mm"].first()
    j = levels.merge(ann, on=["district", "year"], how="inner")
    rows = []
    for district, grp in j.groupby("district"):
        s = grp.dropna(subset=["level_mbgl", "annual_rainfall_mm"])
        if len(s) < 5:
            continue
        corr = s["level_mbgl"].corr(s["annual_rainfall_mm"])
        rows.append({"district": district, "corr_rainfall_level": corr, "n": len(s)})
    return pd.DataFrame(rows).sort_values("corr_rainfall_level")


def fig_national_trend(trend: pd.DataFrame) -> go.Figure:
    trend = trend.copy()
    trend["period"] = trend["year"].astype(str) + " " + trend["season"]
    fig = px.line(
        trend, x="period", y="level_mbgl", markers=True,
        labels={"level_mbgl": "Mean depth below ground (m)", "period": "Season"},
        title="National mean groundwater depth by season",
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_district_series(levels: pd.DataFrame, district: str) -> go.Figure:
    d = levels[levels["district"] == district].sort_values("date")
    fig = go.Figure()
    for season, colour in zip(SEASON_ORDER, ("#d62728", "#1f77b4")):
        s = d[d["season"] == season]
        fig.add_scatter(
            x=s["date"], y=s["level_mbgl"], mode="lines+markers", name=season,
            line=dict(color=colour),
        )
    fig.update_layout(
        title=f"Groundwater depth — {district}",
        xaxis_title="Date", yaxis_title="Depth below ground (m)",
        height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_rainfall_levels(rainfall: pd.DataFrame, levels: pd.DataFrame, district: str) -> go.Figure:
    ann = rainfall.groupby(["district", "year"], as_index=False)["annual_rainfall_mm"].first()
    d = levels[levels["district"] == district].groupby("year", as_index=False)["level_mbgl"].mean()
    j = d.merge(ann, on="year", how="inner").sort_values("year")
    fig = go.Figure()
    fig.add_bar(x=j["year"], y=j["annual_rainfall_mm"], name="Annual rainfall (mm)", opacity=0.5)
    fig.add_scatter(
        x=j["year"], y=j["level_mbgl"], name="Mean GW depth (m)",
        mode="lines+markers", yaxis="y2", line=dict(color="#d62728"),
    )
    fig.update_layout(
        title=f"Rainfall vs groundwater — {district}",
        yaxis=dict(title="Rainfall (mm)"),
        yaxis2=dict(title="GW depth (m)", overlaying="y", side="right"),
        height=380, margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def fig_hotspots(decline: pd.DataFrame, top_n: int = 15) -> go.Figure:
    d = decline.head(top_n).iloc[::-1]
    fig = px.bar(
        d, x="decline_m_per_year", y="district", orientation="h",
        color="decline_m_per_year", color_continuous_scale="Reds",
        labels={"decline_m_per_year": "Decline (m/yr)", "district": ""},
        title=f"Top {top_n} depleting districts",
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
    return fig
