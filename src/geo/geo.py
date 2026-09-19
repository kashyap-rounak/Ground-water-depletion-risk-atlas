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

    Matching order: exact (district+state) -> exact district -> fuzzy district.
    """
    import geopandas as gpd

    p = predictions.copy()
    p["district_norm"] = p["district"].map(normalize_name)
    p["state_norm"] = p["state"].map(normalize_name)

    g1 = gdf.merge(p, on=["district_norm", "state_norm"], how="inner", suffixes=("_geo", ""))
    matched_names = set(g1["district_norm"])
    remaining = p[~p["district_norm"].isin(matched_names)]

    if len(remaining):
        g2 = gdf.merge(
            remaining.drop(columns=["state_norm"]),
            on="district_norm", how="inner", suffixes=("_geo", ""),
        )
        matched_names |= set(g2["district_norm"])
        remaining = p[~p["district_norm"].isin(matched_names)]

    if len(remaining):  # fuzzy fallback on the district name alone
        import difflib
        choices = gdf["district_norm"].unique().tolist()
        fuzz_rows = []
        for _, row in remaining.iterrows():
            match = difflib.get_close_matches(row["district_norm"], choices, n=1, cutoff=0.8)
            if match:
                sub = gdf[gdf["district_norm"] == match[0]]
                row = row.copy()
                row["geometry"] = sub.geometry.iloc[0]
                row["NAME_2"] = sub["NAME_2"].iloc[0]
                fuzz_rows.append(row)
        if fuzz_rows:
            g3 = gpd.GeoDataFrame(pd.DataFrame(fuzz_rows), geometry="geometry", crs=gdf.crs)
            g1 = pd.concat([g1, g2, g3], ignore_index=True)
            matched_names |= set(g3["district_norm"])
        else:
            g1 = pd.concat([g1, g2], ignore_index=True)

    unmatched = p[~p["district_norm"].isin(matched_names)]
    centroids = None
    if len(g1):
        cent = g1.geometry.centroid
        centroids = pd.DataFrame(
            {
                "district": g1["district"].values,
                "state": g1["state"].values,
                "latitude": cent.y.values,
                "longitude": cent.x.values,
            }
        )
    return g1, unmatched, centroids


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
    props = ["district", "state", "risk_label", "risk_score", "level_mbgl",
             "trend_slope", "predicted_decline_m_per_year"]
    out = merged.copy()
    out["key"] = out["district_norm"]
    out["state"] = out.get("state", out.get("state_norm"))
    cols = ["key", "geometry"] + [c for c in props if c in out.columns]
    out = out[cols]
    path = Path(art_dir) / RISK_GEOJSON
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(path, driver="GeoJSON")
    return path
