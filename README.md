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

## 🧠 Tech stack

Python · NumPy · Pandas · Scikit-learn · XGBoost · Statsmodels (SARIMA) · SHAP ·
GeoPandas · Plotly · Streamlit · PyYAML

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

> 💡 **Golden rule:** if a command says `python is not recognized` or `No module named
> streamlit`, you forgot step 2 — activate the venv first.

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
  numpy, pandas, pyarrow, plotly, streamlit, pyyaml) and reads the tracked
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

## Public India data used for this atlas run

The nationwide observational run uses public extracts hosted by the India Data Portal:

- **Groundwater:** [CGWB — Changes in Depth to Water Level](https://ckandev.indiadataportal.com/dataset/8ceb3cea-a8a9-4624-af75-fc92429d8fa5/resource/580a8f6e-3d86-4ca7-ac7d-cd5df12b443c).
  The portal attributes the source to the Central Ground Water Board / India-WRIS.
- **Rainfall:** [Daily Rainfall Data — District Level](https://ckandev.indiadataportal.com/dataset/25664936-d131-4b07-845e-4bd2c0d794aa/resource/d97707e6-9309-46a5-ac31-e8482ce8f9eb).
  The portal attributes the source to India-WRIS and describes the actual-rainfall
  input as a compilation from IMD gridded rainfall, NRSC, APWRIMS, and other agencies.
- Both dataset packages declare the **Open Data Commons Attribution License
  (ODC-BY)**. Refer to each source page for current terms and metadata.

These are India Data Portal extracts, not direct downloads from CGWB or IMD. Their
observation dates and completeness differ. Daily district rainfall is reduced to
pre-/post-monsoon totals in chunks; full-year rainfall sums are preserved separately.
Groundwater time-series and joins are scoped by state as well as district. The raw
CSVs are intentionally excluded from Git; keep them in `data/raw/` when reproducing
this run. Run metadata records the groundwater observation range and provenance.

### Coverage and interpretation

- Source coverage: groundwater observations span **2013-01-01–2023-03-31**;
  rainfall spans **2009-01-01–2024-07-31**. The latest groundwater seasons are very
  sparse: the processed series contain only 10 district-seasons in 2021, 9 in 2022,
  and 3 in 2023, compared with 532 in 2020. Forecasts therefore have different
  effective last-observation dates across districts.
- Only the configured pre-monsoon (March–May) and post-monsoon (October–December)
  groundwater readings enter seasonal modeling. **281,876 of 547,709 valid
  groundwater observations (51.5%) were outside those windows** and excluded;
  January/August readings are not reclassified as monsoon seasons.
- Exact state-and-district name joins matched **539 of 567** groundwater pairs to
  rainfall pairs. Unmatched districts remain in groundwater analysis without a
  rainfall match; names are not silently fuzzy-joined for this calculation.
- Rainfall values are a portal compilation of underlying sources, not uniform direct
  station measurements. District aggregation and varying source coverage limit
  fine-grained interpretation.

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
requirements-app.txt         # slim deps for the cloud deploy (dashboard only)
.streamlit/config.toml       # dashboard theme

data/raw|geo                 # raw data + boundary cache (gitignored; geo auto-downloads)
data/processed/*.parquet     # cleaned tables — TRACKED (the cloud app reads these)
models/artifacts             # trained models, forecasts, metrics — TRACKED
reports/figures              # risk_map.geojson — TRACKED (EDA html is gitignored)
```

## Metrics snapshot (nationwide observational run)

Pipeline run: **2026-10-10**, on the local portal extracts described above, with
synthetic fallback disabled. These are historical backtest results on this dataset,
not a guarantee of performance in future periods or other data sources.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (chronological holdout) | 1.790 m | 2.718 m | 0.8684 |
| Persistence baseline (holdout MAE) | 1.907 m | — | — |
| XGBoost (walk-forward CV mean) | 2.088 m | 3.372 m | 0.7137 |
| SARIMA (sampled backtest) | 2.032 m | 2.771 m | -0.4613 |

The risk classifier's reported **99% agreement** is agreement with labels generated
from threshold rules that also use overlapping model/trend signals; it is not
independent validation of future risk prediction. SARIMA's negative R² indicates it
did not beat the holdout mean baseline on its sampled backtest and should not be
presented as a reliable forecast model without further evaluation.

## Notes & assumptions

- Future rainfall in the XGBoost recursive forecast is assumed equal to each district's
  trailing 3-year mean (documented in `predict_future_levels`).
- Risk labels are rule-based from depletion signals; the classifier learns to reproduce and
  generalise them, and SHAP explains its decisions.
- District ↔ polygon matching prefers district-and-state matches, handles known
  spelling aliases, and only uses unambiguous state-constrained fallbacks. Unmatched
  districts are reported and use station centroids where coordinates are available.
- Completed pipeline runs write `models/artifacts/run_metadata.json` with the data
  source, groundwater observation date range, source date ranges, and any recorded
  provenance. Raw extracts are not included in Git, so other users must acquire the
  same licensed source files to reproduce this run.
