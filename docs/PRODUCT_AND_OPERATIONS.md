# Product and operations guide

## 1. Problem

Groundwater readings are collected at monitoring stations and differ by season,
coverage, and reporting format. The atlas makes those observations easier to
compare by aggregating station readings to district-season summaries and pairing
them with rainfall history. It is a screening and communication tool; it does not
measure aquifer storage, establish sustainable yield, or prescribe extraction.

## 2. User personas

| Persona | Primary need | Atlas workflow |
|---|---|---|
| Water-resource analyst | Find districts that merit data review or additional monitoring | Risk map, district history, rainfall comparison, CSV export |
| District planner or field coordinator | Explain local trends and plan a follow-up assessment | District Analysis and forecast intervals |
| Researcher or data steward | Reproduce transformations and assess model limitations | Configured data ingestion, pipeline artifacts, metrics, API |
| Presenter or program lead | Communicate what the evidence does—and does not—support | Demo & Decision Brief and model-performance view |

## 3. Features

The pipeline accepts configured local CSV/Excel sources or generated demo data;
normalizes common schemas (including year-by-season CGWB-style level columns);
cleans and aggregates observations; derives lag, rainfall, and trend features;
forecasts with SARIMA and XGBoost; assigns transparent threshold-based risk
classes; and publishes district-level maps and tables. The Streamlit app provides
the interactive atlas and presentation narrative. A read-only API exposes
published district summaries and district-season histories.

## 4. System architecture

```text
CSV / Excel or synthetic demo generator
                 |
        schema normalization
                 |
     cleaning and validation
                 |
  district-season aggregation ---- station coordinates -> centroid fallback
                 |
       feature engineering
          /             \
 SARIMA forecast       XGBoost forecast
          \             /
       threshold risk labels -> classifier surrogate + SHAP
                 |
       Parquet + JSON artifacts
          /                 \
 Streamlit dashboard     FastAPI read-only service
```

`scripts/run_pipeline.py` is the batch entry point. The UI and API read the same
generated files; they do not retrain models per request. District geometries are
optional and cached separately. The app falls back to station/polygon centroids
when boundary data is unavailable.

## 5. Data and storage design

Raw observations remain local to the configured pipeline unless the operator
chooses to store them elsewhere. The pipeline writes these canonical tables:

| Dataset | Grain | Key fields |
|---|---|---|
| `station_clean.parquet` | Station × observation date | district, state, station_id, date, level_mbgl, rainfall_mm, latitude, longitude |
| `district_levels.parquet` | District × year × season | district, state, date, season, level_mbgl, n_stations |
| `district_rainfall.parquet` | District × year × season | district, year, season, rainfall_mm, annual_rainfall_mm |
| `features.parquet` | District × season | groundwater lags, target, rainfall windows, seasonal flag, trend |
| `district_summary.parquet` | District | latest depth, rainfall, historical/forecast decline, risk_score, risk_label |

`models/artifacts/atlas_manifest.json` records schema version, UTC generation
time, the data source actually used (including synthetic fallback), and row
counts. It deliberately omits local file paths and station identifiers. Parquet
tables and forecasts are the durable analytical outputs; the manifest is their
run-level provenance record. Re-run the pipeline after changing source data or
configuration.

### API contract

Base path: `/api/v1`. Responses are JSON; all endpoints are read-only.

| Method and path | Purpose |
|---|---|
| `GET /api/v1/health` | Artifact readiness, district/observation counts, and available run provenance |
| `GET /api/v1/summary/risk` | District totals by Low, Moderate, High, and Very High |
| `GET /api/v1/districts?state=&risk=&limit=100&offset=0` | Filtered and paginated district summaries; `limit` is 1–500 |
| `GET /api/v1/districts/{district}/series?state=` | Ordered district-season history; ambiguous names require `state` |

Missing or invalid published datasets return `503`; invalid query values return
`422`; unknown districts return `404`; ambiguous district names return `409`.
The service never accepts a filesystem path, data upload, write, or model
operation from a caller. Interactive OpenAPI documentation is available at
`/docs` when the service is running.

## 6. AI/ML component and interpretation

* SARIMA models each district's biannual series and returns a 95% forecast
  interval. Districts with too few observations may be omitted.
* XGBoost predicts next-season depth and recursively builds a multi-year path.
  Future rainfall is assumed to equal the trailing three-year mean.
* Risk categories are rule-derived from the larger of historical and forecast
  depth slopes using the thresholds in `config.yaml`. A deeper depth below
  ground is not synonymous with aquifer storage or water availability.
* The risk classifier is a surrogate for those rule-derived categories, not a
  separately observed ground-truth outcome. Its report measures agreement with
  generated labels and should not be presented as field validation. The
  classifier excludes the trend and forecast signals that directly define its
  labels. SHAP describes model behavior, not causality.

Use the time-ordered regression holdout and persistence baseline for model
comparison. Review the sample size, coverage, interval width, and local
measurement quality before drawing conclusions; synthetic-data metrics are
demonstration metrics only.

## 7. Security and threat model

See [SECURITY.md](../SECURITY.md) for assets, trust boundaries, threat scenarios,
controls, and deployment guidance. The central boundary is that locally supplied
station files are not exposed by the public read-only API; only district-level
aggregates are served.

## 8. UI/UX and accessibility

The app starts with a demo/decision brief, then supports map-to-district
exploration, risk ranking, forecast comparison, rainfall context, and model
inspection. State and district selectors are shared navigation controls.
Downloadable CSV output supports offline analysis. Risk is encoded with both
color and text labels. Map tiles and boundary availability are environmental
dependencies; the centroid map remains available when polygons are absent.

## 9. Implementation

Install `requirements.txt` for the pipeline, tests, UI, and API. Configure
`config.yaml` to choose synthetic or local inputs, column aliases, cleaning
limits, model parameters, and artifact paths. Keep raw station files and local
credentials outside version control. The batch pipeline is repeatable and
overwrites generated artifacts only when explicitly run.

## 10. Testing

Run `python -m pytest tests -q`. The suite covers cleaning/aggregation/features,
year-and-season wide-file normalization, and API filtering, pagination,
ambiguity handling, response serialization, and read-only behavior. For a
release, also run the full pipeline on representative, non-sensitive input and
review its provenance manifest and forecast metrics.

## 11. Deployment

The Streamlit Community Cloud deployment remains the interactive dashboard.
To run the API locally, install the project dependencies and execute:

```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

For a network deployment, place the API behind a TLS-terminating reverse proxy,
bind only to the private service interface, and apply request-rate limits at
that proxy. Publish only generated, district-level artifacts. Rebuild or restart
the service after a pipeline run to ensure API and dashboard consumers use the
same artifact generation.

## 12. Demo and pitch

**Pitch:** “The Groundwater Depletion Risk Atlas turns fragmented station and
rainfall observations into a transparent district-level screening view. Analysts
can trace seasonal history, compare rainfall, and review short-horizon model
forecasts to prioritize where monitoring and local assessment may be needed. It
is a reproducible decision-support aid—not an automated verdict on water
availability or extraction.”

**Three-minute walkthrough:** start with the Demo & Decision Brief; show the
India map and state filter; open a district to compare groundwater and rainfall;
compare the two forecast paths and uncertainty interval; finish with the at-risk
CSV export and explain the rule thresholds and validation caveats.
