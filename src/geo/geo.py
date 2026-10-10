"""Geospatial helpers: district boundary loading and name matching.

Downloads India district boundaries (GADM-derived GeoJSON) once and caches
them in data/geo/. If unavailable, the dashboard falls back to a lat/lon
centroid map so it always renders.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BOUNDARIES_FILE = "india_districts.geojson"
CENTROIDS_CSV = "district_centroids.csv"
RISK_GEOJSON = "risk_map.geojson"

# normalised old-name -> new-name aliases (GADM predates some renames)
_ALIASES = {
    "bangalorerural": "bengalururural",
    "bangaloreurban": "bengaluruurban",
    "mysore": "mysuru",
    "nasik": "nashik",
    "sholapur": "solapur",
    "guwahati": "kamrupmetropolitan",
    "bhubaneswar": "khordha",
    "pondicherry": "puducherry",
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "nctofdelhi": "delhi",
    "dadraandnagarhaveli": "dadranagarhaveli",
}


def normalize_name(name) -> str:
    """Lowercase, strip accents/brackets/punctuation for robust matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z]", "", s)
    return _ALIASES.get(s, s)


def feature_key(district: str, state: str) -> str:
    """Build a map key that remains unique when district names repeat."""
    return f"{normalize_name(state)}__{normalize_name(district)}"


def combine_centroids(
    polygon_centroids: pd.DataFrame | None,
    station_centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer polygon centroids while retaining station points for unmatched areas."""
    if polygon_centroids is None or polygon_centroids.empty:
        return station_centroids.copy()
    if station_centroids.empty:
        return polygon_centroids.copy()
    return (
        pd.concat([polygon_centroids, station_centroids], ignore_index=True)
        .drop_duplicates(["district", "state"], keep="first")
        .reset_index(drop=True)
    )


def download_boundaries(url: str, geo_dir: Path) -> Path:
    """Fetch the districts GeoJSON once; return the cached file path."""
    geo_dir = Path(geo_dir)
    geo_dir.mkdir(parents=True, exist_ok=True)
    out = geo_dir / BOUNDARIES_FILE
    if out.exists() and out.stat().st_size > 10_000:
        return out
    import requests

    logger.info("Downloading district boundaries from %s", url)
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return out


def load_boundaries(cfg):
    """Load district polygons (GeoDataFrame) or None when offline/broken."""
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas missing; centroid fallback will be used.")
        return None
    try:
        path = download_boundaries(cfg.geo.get("boundaries_url"), Path(cfg.geo_dir))
        gdf = gpd.read_file(path)
        gdf["district_norm"] = gdf["NAME_2"].map(normalize_name)
        gdf["state_norm"] = gdf["NAME_1"].map(normalize_name)
        return gdf[["NAME_2", "NAME_1", "district_norm", "state_norm", "geometry"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load district boundaries: %s", exc)
        return None


def match_predictions(
    predictions: pd.DataFrame, gdf
):
    """Join predictions to polygons; return (merged, unmatched, centroids).

    Matching order: exact district+state, an exact district name only when the
    input state is absent, then an unambiguous fuzzy match within the same state.
    """
    required_predictions = {"district", "state"}
    required_boundaries = {"district_norm", "state_norm", "geometry", "NAME_2"}
    missing_predictions = required_predictions.difference(predictions.columns)
    missing_boundaries = required_boundaries.difference(gdf.columns)
    if missing_predictions or missing_boundaries:
        missing = sorted(missing_predictions | missing_boundaries)
        raise ValueError(f"Missing geospatial match columns: {', '.join(missing)}")

    import geopandas as gpd
    from difflib import SequenceMatcher, get_close_matches

    p = predictions.copy()
    p["district_norm"] = p["district"].map(normalize_name)
    p["state_norm"] = p["state"].map(normalize_name)
    p["_prediction_id"] = range(len(p))

    matched = gdf.merge(
        p,
        on=["district_norm", "state_norm"],
        how="inner",
        suffixes=("_geo", ""),
    )
    matched_ids = set(matched["_prediction_id"])
    unresolved = p[~p["_prediction_id"].isin(matched_ids)]
    additional_matches = []

    for _, row in unresolved.iterrows():
        same_name = gdf[gdf["district_norm"] == row["district_norm"]]
        same_state = same_name[same_name["state_norm"] == row["state_norm"]]
        candidates = same_state

        if candidates.empty and not row["state_norm"] and same_name["state_norm"].nunique() == 1:
            candidates = same_name

        if candidates.empty:
            state_boundaries = gdf[gdf["state_norm"] == row["state_norm"]]
            choices = state_boundaries["district_norm"].dropna().unique().tolist()
            close = get_close_matches(row["district_norm"], choices, n=2, cutoff=0.8)
            if close:
                best_score = SequenceMatcher(None, row["district_norm"], close[0]).ratio()
                second_score = (
                    SequenceMatcher(None, row["district_norm"], close[1]).ratio()
                    if len(close) > 1 else 0.0
                )
                if len(close) == 1 or best_score - second_score >= 0.05:
                    candidates = state_boundaries[
                        state_boundaries["district_norm"] == close[0]
                    ]
                else:
                    logger.warning(
                        "Ambiguous fuzzy boundary match for %s, %s; leaving unmatched.",
                        row["district"],
                        row["state"],
                    )

        if not candidates.empty:
            # A district may have multipart boundary records; retain all of
            # them, but never select a geometry from a different state.
            candidate_rows = candidates.copy()
            for column in p.columns:
                if column not in candidate_rows.columns:
                    candidate_rows[column] = row[column]
            candidate_rows["district_norm"] = row["district_norm"]
            candidate_rows["state_norm"] = row["state_norm"]
            additional_matches.append(candidate_rows)

    if additional_matches:
        matched = pd.concat(
            [matched, *additional_matches],
            ignore_index=True,
        )
    matched = gpd.GeoDataFrame(matched, geometry="geometry", crs=gdf.crs)
    matched_ids = set(matched["_prediction_id"])
    unmatched = p[~p["_prediction_id"].isin(matched_ids)].drop(columns="_prediction_id")

    centroids = None
    if len(matched):
        if matched.crs is None:
            logger.warning(
                "Boundary CRS is missing; using representative points for map locations."
            )
            cent = matched.geometry.representative_point()
        else:
            cent = (
                matched.to_crs("EPSG:6933")
                .geometry.centroid
                .to_crs(matched.crs)
            )
        centroids = pd.DataFrame(
            {
                "district": matched["district"].values,
                "state": matched["state"].values,
                "latitude": cent.y.values,
                "longitude": cent.x.values,
            }
        )
    return matched, unmatched, centroids


def build_centroids(cfg, levels: pd.DataFrame) -> pd.DataFrame:
    """Lat/lon per district from observed station coordinates (fallback map)."""
    c = (
        levels.dropna(subset=["latitude", "longitude"])
        .groupby(["district", "state"], as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    )
    out = Path(cfg.geo_dir) / CENTROIDS_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(out, index=False)
    return c


def save_risk_geojson(merged, art_dir: Path) -> Path | None:
    """Write merged risk GeoJSON with a plotly-friendly `key` property."""
    if merged is None or not len(merged):
        return None
    props = [
        "district", "state", "state_norm", "risk_label", "risk_score",
        "level_mbgl", "trend_slope", "predicted_decline_m_per_year",
    ]
    out = merged.copy()
    out["key"] = [
        feature_key(district, state)
        for district, state in zip(out["district"], out["state"])
    ]
    out["key_scope"] = "state_district_v1"
    out["state"] = out.get("state", out.get("state_norm"))
    cols = ["key", "geometry"] + [c for c in props if c in out.columns]
    out = out[cols]
    path = Path(art_dir) / RISK_GEOJSON
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(path, driver="GeoJSON")
    return path
