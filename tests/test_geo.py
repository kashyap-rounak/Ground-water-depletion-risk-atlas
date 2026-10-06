"""Tests for map coordinate fallback and polygon centroid projection."""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.config import load_config
from src.geo.geo import build_centroids, match_predictions


def test_station_centroids_are_saved_as_district_fallback(tmp_path):
    cfg = load_config()
    cfg.raw.setdefault("paths", {})["geo"] = str(tmp_path)
    stations = pd.DataFrame([
        {"district": "Alpha", "state": "North", "latitude": 22.0, "longitude": 80.0},
        {"district": "Alpha", "state": "North", "latitude": 24.0, "longitude": 82.0},
    ])

    centroids = build_centroids(cfg, stations)

    assert centroids.iloc[0]["latitude"] == 23.0
    assert centroids.iloc[0]["longitude"] == 81.0
    assert (tmp_path / "district_centroids.csv").is_file()


def test_polygon_centroids_are_projected_before_conversion():
    boundaries = gpd.GeoDataFrame(
        {
            "NAME_2": ["Alpha"],
            "NAME_1": ["North"],
            "district_norm": ["alpha"],
            "state_norm": ["north"],
        },
        geometry=[Polygon([(79.0, 21.0), (81.0, 21.0), (81.0, 23.0), (79.0, 23.0)])],
        crs="EPSG:4326",
    )
    predictions = pd.DataFrame([
        {"district": "Alpha", "state": "North", "risk_label": "Low"},
    ])

    _, unmatched, centroids = match_predictions(predictions, boundaries)

    assert unmatched.empty
    assert abs(centroids.iloc[0]["latitude"] - 22.0) < 0.01
    assert abs(centroids.iloc[0]["longitude"] - 80.0) < 0.01
