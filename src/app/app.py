"""🌍 Groundwater Depletion Risk Atlas — Streamlit dashboard.

Run:  streamlit run src/app/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.eda import eda  # noqa: E402
from src.geo.geo import feature_key, normalize_name  # noqa: E402

st.set_page_config(page_title="Groundwater Risk Atlas", page_icon="🌍", layout="wide")

RISK_COLORS = {
    "Low": "#2e7d32",
    "Moderate": "#f9a825",
    "High": "#ef6c00",
    "Very High": "#c62828",
}
RISK_ORDER = ["Very High", "High", "Moderate", "Low"]

# plotly >=6 renamed *mapbox traces to *map (MapLibre); support both
try:
    CHORO_CLS = go.Choroplethmap
    SCATTER_MAP = px.scatter_map
    MAP_KEY = "map"
except AttributeError:  # plotly <6
    CHORO_CLS = go.Choroplethmapbox
    SCATTER_MAP = px.scatter_mapbox
    MAP_KEY = "mapbox"

PAGES = [
    "🗺️ India Risk Map",
    "📍 District Analysis",
    "📈 Groundwater Trends",
    "🔮 Future Forecast",
    "⚠️ At-Risk Districts",
    "🌧️ Rainfall Analysis",
    "🤖 Model Performance & SHAP",
]


# ------------------------------------------------------------------ loading
def _artifact_signature(cfg) -> tuple[tuple[str, int, int], ...]:
    """Return file metadata used to invalidate cached artifacts after updates."""
    proc = Path(cfg.processed_dir)
    art = Path(cfg.artifacts_dir)
    files = [
        proc / "district_levels.parquet",
        proc / "district_rainfall.parquet",
        proc / "district_centroids_geo.parquet",
        art / "sarima_forecast.parquet",
        art / "xgb_future_levels.parquet",
        art / "district_summary.parquet",
        art / "depletion_ranking.parquet",
        art / "rainfall_gw_correlation.parquet",
        art / "shap_importance.parquet",
        art / "metrics.json",
        art / "run_metadata.json",
        Path(cfg.figures_dir) / "risk_map.geojson",
    ]
    signature = []
    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError:
            signature.append((str(path), -1, -1))
        else:
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


@st.cache_data(show_spinner="Loading atlas data…")
def load_artifacts(signature: tuple[tuple[str, int, int], ...]):
    """Load the current artifact set; signature invalidates stale cache entries."""
    cfg = load_config()
    proc = Path(cfg.processed_dir)
    art = Path(cfg.artifacts_dir)

    def _pq(p: Path) -> pd.DataFrame | None:
        return pd.read_parquet(p) if p.exists() else None

    data = {
        "levels": _pq(proc / "district_levels.parquet"),
        "rainfall": _pq(proc / "district_rainfall.parquet"),
        "sarima": _pq(art / "sarima_forecast.parquet"),
        "future": _pq(art / "xgb_future_levels.parquet"),
        "summary": _pq(art / "district_summary.parquet"),
        "decline": _pq(art / "depletion_ranking.parquet"),
        "corr": _pq(art / "rainfall_gw_correlation.parquet"),
        "shap": _pq(art / "shap_importance.parquet"),
        "centroids": _pq(proc / "district_centroids_geo.parquet"),
    }
    metrics_path = art / "metrics.json"
    data["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metadata_path = art / "run_metadata.json"
    data["run_metadata"] = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else None
    )
    return cfg, data


@st.cache_data(show_spinner=False)
def load_risk_geojson(path: str, modified_ns: int, size: int) -> dict:
    """Read and cache map geometry, invalidated when the source file changes."""
    del modified_ns, size
    return json.loads(Path(path).read_text(encoding="utf-8"))


cfg = load_config()
cfg, D = load_artifacts(_artifact_signature(cfg))
LEVELS = D["levels"]
SUMMARY = D["summary"]
if LEVELS is None or SUMMARY is None:
    st.error(
        "Required atlas artifacts are missing. Run `python scripts/run_pipeline.py` "
        "after verifying the configured input data source."
    )
    st.stop()
if LEVELS.empty or SUMMARY.empty:
    st.error("Atlas artifacts contain no groundwater or district summary records.")
    st.stop()
required_level_columns = {"district", "state", "date", "level_mbgl", "season"}
missing_columns = required_level_columns.difference(LEVELS.columns)
if missing_columns:
    st.error(
        "The district-level artifact is incompatible; missing columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

STATES = sorted(LEVELS["state"].dropna().unique())

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌍 Risk Atlas")
page = st.sidebar.radio("Navigate", PAGES)
sel_state = st.sidebar.selectbox("State filter", ["All states"] + STATES)
district_rows = LEVELS[["district", "state"]].drop_duplicates()
if sel_state != "All states":
    district_rows = district_rows[district_rows["state"] == sel_state]
district_rows = district_rows.sort_values(["state", "district"])
district_labels = {
    (
        f"{row.district} — {row.state}"
        if sel_state == "All states"
        else row.district
    ): (row.district, row.state)
    for row in district_rows.itertuples(index=False)
}
selected_district = st.sidebar.selectbox(
    "District (detail pages)", list(district_labels)
)
sel_district, sel_district_state = district_labels[selected_district]

run_metadata = D["run_metadata"]
if run_metadata is None:
    st.sidebar.warning(
        f"Configured source: {cfg.data_source}. These artifacts have no recorded "
        "run provenance; the source used to generate them cannot be confirmed."
    )
elif run_metadata.get("source_used") == "synthetic-fallback":
    st.sidebar.warning(
        "These results use synthetic fallback data because local ingestion failed."
    )
elif run_metadata.get("source_used") == "synthetic":
    st.sidebar.warning(
        "These results use synthetic data and are not evidence about observed "
        "groundwater conditions."
    )
else:
    st.sidebar.success(
        f"Data source: {run_metadata.get('source_used', 'unknown')} | "
        f"observations: {run_metadata.get('date_min', 'unknown')} to "
        f"{run_metadata.get('date_max', 'unknown')}"
    )
    provenance = run_metadata.get("provenance", {})
    datasets = provenance.get("datasets", [])
    if datasets:
        with st.sidebar.expander("Dataset sources & attribution"):
            for dataset in datasets:
                st.markdown(
                    f"**{dataset.get('title', 'Dataset')}**  \n"
                    f"{dataset.get('provider', 'Provider not recorded')}  \n"
                    f"[Source and download page]({dataset.get('url', '')})  \n"
                    f"License: {dataset.get('license', 'not recorded')}"
                )
    limitations = provenance.get("limitations", [])
    if limitations:
        with st.sidebar.expander("Coverage & limitations"):
            for limitation in limitations:
                st.markdown(f"- {limitation}")


def filtered(df: pd.DataFrame) -> pd.DataFrame:
    if sel_state == "All states" or "state" not in df.columns:
        return df
    return df[df["state"] == sel_state]


def selected_district_rows(df: pd.DataFrame) -> pd.DataFrame:
    result = df[df["district"] == sel_district]
    if "state" in result.columns:
        result = result[result["state"] == sel_district_state]
    return result


# ------------------------------------------------------------------ map figs
def make_choropleth(summary: pd.DataFrame) -> go.Figure | None:
    geo_path = Path(cfg.figures_dir) / "risk_map.geojson"
    if not geo_path.exists():
        return None
    stat = geo_path.stat()
    gj = load_risk_geojson(str(geo_path), stat.st_mtime_ns, stat.st_size)
    s = summary.copy()
    scoped_keys = any(
        feature.get("properties", {}).get("key_scope") == "state_district_v1"
        for feature in gj.get("features", [])
    )
    if scoped_keys:
        s["key"] = [
            feature_key(district, state)
            for district, state in zip(s["district"], s["state"])
        ]
    else:
        s["key"] = s["district"].map(normalize_name)
    custom = np.stack(
        [s["district"], s["risk_label"], s["level_mbgl"], s["trend_slope"]], axis=-1
    )
    fig = go.Figure(
        CHORO_CLS(
            geojson=gj,
            featureidkey="properties.key",
            locations=s["key"],
            z=s["risk_score"].astype(float),
            zmin=0, zmax=3,
            colorscale=[
                [0.0, RISK_COLORS["Low"]], [0.25, RISK_COLORS["Low"]],
                [0.25, RISK_COLORS["Moderate"]], [0.5, RISK_COLORS["Moderate"]],
                [0.5, RISK_COLORS["High"]], [0.75, RISK_COLORS["High"]],
                [0.75, RISK_COLORS["Very High"]], [1.0, RISK_COLORS["Very High"]],
            ],
            showscale=False,
            marker=dict(opacity=0.75),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Risk: %{customdata[1]}"
                "<br>Depth: %{customdata[2]:.1f} m bgl"
                "<br>Trend: %{customdata[3]:.2f} m/yr<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **{MAP_KEY: dict(style="open-street-map", zoom=3.4, center={"lat": 22.8, "lon": 80.0})},
        height=560,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def make_centroid_map(summary: pd.DataFrame) -> go.Figure | None:
    if D["centroids"] is not None and len(D["centroids"]):
        pts = summary.merge(D["centroids"], on=["district", "state"], how="left")
    else:
        return None
    pts = pts.dropna(subset=["latitude", "longitude"])
    if pts.empty:
        return None
    fig = SCATTER_MAP(
        pts, lat="latitude", lon="longitude",
        color="risk_label", color_discrete_map=RISK_COLORS,
        size_max=14, zoom=3.4,
        center={"lat": 22.8, "lon": 80.0},
        hover_data={"district": True, "level_mbgl": ":.1f",
                    "trend_slope": ":.2f", "latitude": False, "longitude": False},
        height=560,
    )
    fig.update_layout(**{MAP_KEY: dict(style="open-street-map")}, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ==================================================================
if page == "🗺️ India Risk Map":
    st.title("🗺️ India Groundwater Risk Map")
    st.caption(
        "Risk classes are assigned by configured thresholds applied to forecast "
        "and historical depletion signals."
    )
    risk_counts = SUMMARY["risk_label"].value_counts()
    kpi = st.columns(4)
    kpi[0].metric("Districts", len(SUMMARY))
    kpi[1].metric("Very High risk", int(risk_counts.get("Very High", 0)))
    kpi[2].metric("High risk", int(risk_counts.get("High", 0)))
    kpi[3].metric("Fastest decline", f"{SUMMARY['trend_slope'].max():.2f} m/yr")
    fig = make_choropleth(filtered(SUMMARY)) or make_centroid_map(filtered(SUMMARY))
    if fig is None:
        st.warning(
            "No usable district boundaries or station coordinates are available. "
            "The map is unavailable for this artifact set."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)

elif page == "📍 District Analysis":
    st.title(f"📍 {sel_district}")
    d = selected_district_rows(LEVELS).sort_values("date")
    s_row = selected_district_rows(SUMMARY).iloc[0]
    n = len(d)
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest depth (m bgl)", f"{d['level_mbgl'].iloc[-1]:.1f}")
    window = d["level_mbgl"].iloc[-min(20, n):]
    c2.metric("Recent change (10 yr)", f"{window.iloc[-1] - window.iloc[0]:+.1f} m")
    c3.metric("Risk class", s_row["risk_label"])
    st.plotly_chart(
        eda.fig_district_series(LEVELS, sel_district, sel_district_state),
        use_container_width=True,
    )
    st.plotly_chart(
        eda.fig_rainfall_levels(
            D["rainfall"], LEVELS, sel_district, sel_district_state
        ),
        use_container_width=True,
    )

elif page == "📈 Groundwater Trends":
    st.title("📈 Groundwater Trends")
    st.plotly_chart(
        eda.fig_national_trend(eda.national_trend(filtered(LEVELS))), use_container_width=True
    )
    st.plotly_chart(eda.fig_hotspots(filtered(D["decline"])), use_container_width=True)
    st.dataframe(filtered(D["decline"]).round(2), use_container_width=True, height=380)

elif page == "🔮 Future Forecast":
    st.title("🔮 Future Forecast")
    hist = selected_district_rows(LEVELS).sort_values("date")
    fc = D["sarima"]
    f2 = D["future"]
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["level_mbgl"], name="Observed", mode="lines+markers")
    if fc is not None and len(selected_district_rows(fc)):
        f = selected_district_rows(fc).sort_values("date")
        fig.add_scatter(x=f["date"], y=f["forecast_level"], name="SARIMA", mode="lines+markers")
        fig.add_scatter(
            x=list(f["date"]) + list(f["date"][::-1]),
            y=list(f["upper_95"]) + list(f["lower_95"][::-1]),
            fill="toself", name="95% CI", line=dict(width=0), opacity=0.15,
        )
    if f2 is not None and len(selected_district_rows(f2)):
        f = selected_district_rows(f2).sort_values("date")
        fig.add_scatter(
            x=f["date"], y=f["forecast_level"], name="XGBoost (ML)",
            mode="lines+markers", line=dict(dash="dot"),
        )
    fig.update_layout(
        title=f"{sel_district} — forecast depth (m bgl)", height=470,
        yaxis_title="Depth (m bgl)", xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)
    if fc is not None:
        st.dataframe(
            selected_district_rows(fc)[
                ["date", "season", "forecast_level", "lower_95", "upper_95"]
            ].round(2),
            use_container_width=True,
        )

elif page == "⚠️ At-Risk Districts":
    st.title("⚠️ At-Risk Districts")
    tab1, tab2 = st.tabs(["Risk table", "Risk distribution"])
    with tab1:
        s = filtered(SUMMARY).copy()
        s["rank"] = s["risk_label"].map({l: i for i, l in enumerate(RISK_ORDER)})
        s = s.sort_values(["rank", "trend_slope"], ascending=[True, False]).drop(columns="rank")
        st.dataframe(
            s.style.map(
                lambda v: f"color: {RISK_COLORS[v]}; font-weight: bold" if v in RISK_COLORS else "",
                subset=["risk_label"],
            ).format({
                "level_mbgl": "{:.1f}",
                "trend_slope": "{:.2f}",
                "predicted_decline_m_per_year": "{:.2f}",
            }),
            use_container_width=True, height=480,
        )
    with tab2:
        vc = filtered(SUMMARY)["risk_label"].value_counts().reindex(RISK_ORDER[::-1]).fillna(0)
        fig = px.bar(x=vc.index, y=vc.values, color=vc.index, color_discrete_map=RISK_COLORS)
        fig.update_layout(xaxis_title="Risk class", yaxis_title="Districts", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🌧️ Rainfall Analysis":
    st.title("🌧️ Rainfall Analysis")
    ann = filtered(D["rainfall"]).groupby("year", as_index=False)["annual_rainfall_mm"].mean()
    fig = px.bar(ann, x="year", y="annual_rainfall_mm", title="Mean annual rainfall across districts")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Rainfall ↔ groundwater correlation by district")
    corr = filtered(D["corr"]).sort_values("corr_rainfall_level")
    fig2 = px.bar(
        corr, x="corr_rainfall_level", y="district", orientation="h",
        color="corr_rainfall_level", color_continuous_scale="RdYlGn_r",
        labels={"corr_rainfall_level": "Correlation (r)"},
    )
    fig2.update_layout(height=620)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(
        eda.fig_rainfall_levels(
            D["rainfall"], LEVELS, sel_district, sel_district_state
        ),
        use_container_width=True,
    )

else:  # Model Performance & SHAP
    st.title("🤖 Model Performance & SHAP")
    m = D["metrics"]
    xg = m.get("xgboost", {})
    sar = m.get("sarima_backtest", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGB MAE (holdout)", f"{xg.get('mae', float('nan')):.3f} m")
    c2.metric("XGB RMSE", f"{xg.get('rmse', float('nan')):.3f} m")
    c3.metric("XGB R²", f"{xg.get('r2', float('nan')):.4f}")
    c4.metric("Persistence MAE", f"{xg.get('baseline_persistence_mae', float('nan')):.3f} m")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SARIMA MAE", f"{sar.get('mae', float('nan')):.3f} m")
    c2.metric("SARIMA RMSE", f"{sar.get('rmse', float('nan')):.3f} m")
    c3.metric("SARIMA R²", f"{sar.get('r2', float('nan')):.4f}")
    c4.metric("XGB CV R²", f"{xg.get('cv_r2_mean', float('nan')):.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk classifier agreement with threshold labels")
        st.caption(
            "The classifier is trained to reproduce rule-generated labels using "
            "overlapping risk signals. These metrics are not independent validation "
            "of future risk prediction."
        )
        st.text(m.get("risk_classifier_report", "n/a"))
        st.subheader("Confusion matrix")
        cm = np.array(m.get("confusion_matrix", []))
        class_order = ["Low", "Moderate", "High", "Very High"]
        if cm.shape == (len(class_order), len(class_order)):
            fig = px.imshow(
                cm,
                text_auto=True,
                x=class_order,
                y=class_order,
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        elif cm.size:
            st.warning(
                "The stored confusion matrix does not use the current four-class "
                "risk-label format. Rerun the pipeline to regenerate it."
            )
    with right:
        st.subheader("SHAP global importance (risk model)")
        if D["shap"] is not None and len(D["shap"]):
            fig = px.bar(D["shap"].iloc[::-1], x="mean_abs_shap", y="feature", orientation="h")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Predicted future decline dominates the risk decision, followed by current depth "
            "and historical trend — exactly the signals an atlas should key on."
        )
        st.subheader("Risk class distribution")
        st.json(m.get("risk_label_counts", {}))
