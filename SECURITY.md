# Security policy and threat model

## Scope and assets

The atlas is a local batch pipeline, a Streamlit dashboard, and a read-only
FastAPI service. Protect locally supplied station files (which may contain
precise well coordinates), published analytical artifacts, configuration, and
the integrity of risk/forecast outputs. The API is intended to publish district
aggregates, not station-level data.

## Trust boundaries and threats

| Threat | Impact | Controls and operator responsibilities |
|---|---|---|
| Malformed or unexpectedly large CSV/Excel input | Pipeline failure, resource exhaustion, or bad outputs | Accept files only from trusted operators; inspect provenance and size before processing; review cleaning logs and row counts |
| Sensitive station coordinates exposed through a public endpoint | Privacy or operational-site disclosure | API reads only district summary and district-season tables; no upload, arbitrary path, or station-table endpoint exists; keep raw files and station artifacts private |
| Artifact tampering or stale artifacts | Incorrect risk display or API response | Generate artifacts through the pipeline; review the run manifest and metrics; restrict write access to artifact directories; redeploy/restart after a pipeline run |
| Malicious model pickle substituted into the project | Code execution if untrusted pickle is loaded | Do not load model pickle files from untrusted sources; the public API does not deserialize or use model files |
| Public endpoint abuse | Resource exhaustion or service unavailability | Endpoints are bounded and read-only; deploy behind TLS and a rate-limiting reverse proxy; monitor service and host resource use |
| Misinterpretation of forecasts or risk labels | Inappropriate planning or policy decisions | Show rule thresholds, uncertainty, data-source provenance, model assumptions, and non-causal/non-authoritative limitations in the UI and product guide |
| Dependency or deployment compromise | Application or data integrity loss | Review dependency updates, pin versions for controlled releases, build from trusted sources, and keep deployment credentials outside the repository |

## Security controls

* API routes expose a fixed set of district-level reads only. Pagination is
  limited to 500 rows, string filters are length-bounded, district matching is
  exact and case-insensitive, and no user-supplied path reaches file access.
* The API has no write, upload, model-inference, or station-data route. CORS is
  not enabled by default. Do not add broad CORS or file-serving routes without
  a reviewed requirement.
* Store local input files outside public web roots. Keep private data, tokens,
  and deployment secrets out of source control and logs.
* Use TLS, private networking, least-privilege filesystem access, and a
  rate-limiting reverse proxy for internet-facing deployments. The API does not
  implement user authentication because its intended dataset is district-level
  public decision-support output; do not expose it if deployment policy requires
  authenticated access without first adding that control.
* Treat generated pickle files as trusted build artifacts only. The serving
  application intentionally reads Parquet and JSON outputs instead.

## Reporting

Do not publish sensitive station-level files or secrets in an issue. Report a
suspected vulnerability privately to the repository maintainers with affected
version, impact, and reproduction details.
