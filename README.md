# 🌍 Groundwater Depletion Risk Atlas

An ML-powered web application that analyzes historical groundwater and rainfall data,
predicts future groundwater conditions, classifies depletion risk, visualizes it
geographically, and presents the results through an interactive Streamlit dashboard.

> 🔴 **Live demo:** [ground-water-depletion-risk-atlas.streamlit.app](https://ground-water-depletion-risk-atlas.streamlit.app)

## What it does

1. **Collect** — reads station-level groundwater + rainfall readings (your files, or a
   built-in realistic synthetic generator for demo mode).
2. **Preprocess** — deduplicates, sanity-filters (impossible depths, negative rainfall),
   flags IQR outliers per district, and time-interpolates gaps within each station.
3. **Aggregate** — station readings → district × season (pre-monsoon Mar–May,
   post-monsoon Oct–Dec) master table, plus seasonal/annual rainfall.
4. **EDA** — national & district trends, seasonality, rainfall↔groundwater correlations,
   depletion hotspot ranking.
5. **Feature engineering** — previous-season levels (lags 1/2/4), YoY change, trailing
   3-year rainfall means, rainfall lags, seasonal encodings, trailing trend slope.
6. **Models**
   - **SARIMA** (statsmodels) — seasonal time-series forecast per district (3 years ahead, 95% CI).
   - **XGBoost** — next-season level regression, walk-forward validation, compared against a
     persistence baseline.
   - **Risk classifier** — XGBoost classifier over 4 classes (Low / Moderate / High / Very High)
     with labels derived from predicted decline + historical trend (thresholds in `config.yaml`).
   - **SHAP** — global feature importance + per-prediction attributions for the risk model.
7. **Geospatial** — district polygons (cached GeoJSON) joined by fuzzy-matched names, with a
   station-centroid fallback map when boundaries are unavailable.
8. **Streamlit dashboard** — 7 tabs: 🗺️ India Risk Map · 📍 District Analysis ·
   📈 Groundwater Trends · 🔮 Future Forecast · ⚠️ At-Risk Districts · 🌧️ Rainfall Analysis ·
   🤖 Model Performance & SHAP.

## 🎬 Run for a presentation (models already trained)

If `models/artifacts/` is already populated, you only need the dashboard. Open a
terminal in VS Code (`` Ctrl+` ``) and run these in order:

```powershell
cd "D:\Ground water depletion risk atlas"   # 1. go to the project folder
.venv\Scripts\activate                      # 2. activate the virtual environment
streamlit run src/app/app.py                # 3. launch the dashboard
```

The browser opens automatically at `http://localhost:8501`. To stop after the demo:
click the terminal → `Ctrl+C` → `deactivate`.

To retrain everything first (optional, ~2–3 min, **not** recommended live on stage):

```powershell
python scripts/run_pipeline.py
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/mac)
pip install -r requirements.txt

# 1. run the full pipeline (works immediately on synthetic demo data)
python scripts/run_pipeline.py

# 2. launch the dashboard
.venv\Scripts\streamlit run src/app/app.py
```

Optional: pre-download district boundaries (auto-downloaded on first pipeline run anyway):

```bash
python scripts/download_geo.py
```

Run tests:

```bash
.venv\Scripts\python -m pytest tests -q
```

## ☁️ Deployment (Streamlit Community Cloud)

The app is live at **https://ground-water-depletion-risk-atlas.streamlit.app**.

- **Every `git push` to `main` redeploys the cloud app automatically** — no extra steps.
- The cloud build installs the slim `requirements-app.txt` (dashboard-only dependencies:
  numpy, pandas, plotly, streamlit, pyyaml) and reads the tracked
  `data/processed/*.parquet` + `reports/figures/risk_map.geojson`.
- After retraining with new data, commit the refreshed parquets/GeoJSON and push —
  the live app updates itself.

Sync your changes to GitHub (which also redeploys the cloud app):

```bash
git status                                # see what changed
git add .                                 # stage the changes
git commit -m "describe the change"       # save a checkpoint
git push origin main                      # upload → live app redeploys
```

## 🔑 Using YOUR real data

The pipeline is **path-driven and schema-flexible**:

1. Copy your station-level files (CSV or Excel) into `data/raw/`
   **or** set `data.local_path` in `config.yaml` to any folder/file on your machine.
2. Set `data.source: local` in `config.yaml`.
3. Re-run `python scripts/run_pipeline.py` and refresh the dashboard.

The loader auto-detects your column names via the alias table in `config.yaml`
(`data.column_aliases`) — district, station id, date/year, water level, rainfall,
latitude/longitude all have common spellings pre-mapped. If your headers use a
different spelling, just add it to the alias list. Files in CGWB "wide" style
(separate pre-monsoon / post-monsoon level columns) are automatically converted
to long format.

If local files are missing or unreadable, the pipeline falls back to synthetic data
(`data.allow_synthetic_fallback: true`) so the app never hard-fails.

## Configuration

All knobs live in `config.yaml`: data source/paths, cleaning thresholds, season months,
feature windows, SARIMA/XGBoost hyperparameters, risk thresholds (`risk.decline_m_per_year`
controls the Low→Very-High cutoffs), SHAP sample sizes, and the boundaries URL.

## Project layout

```
config.yaml                  # all configuration
scripts/run_pipeline.py      # ingest → clean → aggregate → features → EDA → models → geo
scripts/download_geo.py      # one-time district boundaries fetch
src/data/                    # ingest, synthetic generator, cleaning, aggregation
src/features/build_features.py
src/eda/eda.py               # figures + statistics (shared with the dashboard)
src/models/                  # sarima, xgboost_model, risk, evaluate, explain (SHAP)
src/geo/geo.py               # boundaries, name matching, centroids
src/app/app.py               # Streamlit dashboard
tests/                       # pytest: cleaning, aggregation, features
data/raw|processed|geo       # data (gitignored)
models/artifacts             # trained models, forecasts, metrics (gitignored)
reports/figures              # saved EDA html + risk_map.geojson
```

## Metrics snapshot (demo data)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (holdout) | 0.379 m | 0.503 m | 0.9987 |
| Persistence baseline | 0.442 m | — | — |
| SARIMA (backtest) | 0.311 m | 0.384 m | 0.9992 |
| Risk classifier accuracy | 93% | | |

## Notes & assumptions

- Future rainfall in the XGBoost recursive forecast is assumed equal to each district's
  trailing 3-year mean (documented in `predict_future_levels`).
- Risk labels are rule-based from depletion signals; the classifier learns to reproduce and
  generalise them, and SHAP explains its decisions.
- District ↔ polygon matching normalises spellings and handles known renames
  (Gurgaon→Gurugram, Mysore→Mysuru, Bangalore Rural→Bengaluru Rural, …), with a fuzzy
  fallback for the rest.
