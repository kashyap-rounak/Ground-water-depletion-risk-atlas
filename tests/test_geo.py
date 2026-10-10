"""Tests for safe district-to-boundary matching decisions."""
from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from src.geo.geo import combine_centroids, feature_key, match_predictions


class _BoundaryFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return _BoundaryFrame

    @property
    def crs(self):
        return "EPSG:4326"


def test_district_name_in_another_state_is_not_matched(monkeypatch):
    geopandas_stub = ModuleType("geopandas")
    geopandas_stub.GeoDataFrame = lambda frame, **kwargs: frame
    monkeypatch.setitem(sys.modules, "geopandas", geopandas_stub)

    predictions = pd.DataFrame(
        {"district": ["Springfield"], "state": ["State A"]}
    )
    boundaries = _BoundaryFrame(
        {
            "NAME_2": ["Springfield"],
            "NAME_1": ["State B"],
            "district_norm": ["springfield"],
            "state_norm": ["stateb"],
            "geometry": [None],
        }
    )

    matched, unmatched, centroids = match_predictions(predictions, boundaries)

    assert matched.empty
    assert unmatched["district"].tolist() == ["Springfield"]
    assert centroids is None


def test_map_feature_keys_include_state_for_repeated_district_names():
    assert feature_key("Springfield", "State A") != feature_key("Springfield", "State B")


def test_polygon_centroids_keep_station_fallback_for_unmatched_districts():
    polygon = pd.DataFrame(
        {
            "district": ["Matched"],
            "state": ["State A"],
            "latitude": [10.0],
            "longitude": [20.0],
        }
    )
    stations = pd.DataFrame(
        {
            "district": ["Matched", "Unmatched"],
            "state": ["State A", "State B"],
            "latitude": [11.0, 30.0],
            "longitude": [21.0, 40.0],
        }
    )

    combined = combine_centroids(polygon, stations).set_index("district")

    assert combined.loc["Matched", "latitude"] == 10.0
    assert combined.loc["Unmatched", "latitude"] == 30.0


def test_boundary_centroid_is_computed_in_projected_crs():
    predictions = pd.DataFrame(
        {"district": ["Test District"], "state": ["Test State"]}
    )
    boundaries = gpd.GeoDataFrame(
        {
            "NAME_2": ["Test District"],
            "NAME_1": ["Test State"],
            "district_norm": ["testdistrict"],
            "state_norm": ["teststate"],
        },
        geometry=[box(70.0, 10.0, 80.0, 30.0)],
        crs="EPSG:4326",
    )

    _, unmatched, centroids = match_predictions(predictions, boundaries)

    assert unmatched.empty
    assert centroids is not None
    assert abs(centroids.loc[0, "longitude"] - 75.0) < 0.1
    assert abs(centroids.loc[0, "latitude"] - 20.0) < 0.5
