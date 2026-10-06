"""Contract tests for the read-only public API."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app as api_module


@pytest.fixture
def client(monkeypatch):
    summary = pd.DataFrame([
        {
            "district": "Alpha", "state": "North", "risk_label": "High",
            "risk_score": 2, "trend_slope": 0.42, "level_mbgl": 18.2,
            "predicted_decline_m_per_year": 0.35,
        },
        {
            "district": "Beta", "state": "South", "risk_label": "Low",
            "risk_score": 0, "trend_slope": 0.02, "level_mbgl": 4.0,
            "predicted_decline_m_per_year": 0.01,
        },
        {
            "district": "Springfield", "state": "East", "risk_label": "Moderate",
            "risk_score": 1, "trend_slope": 0.2, "level_mbgl": float("nan"),
            "predicted_decline_m_per_year": 0.18,
        },
        {
            "district": "Springfield", "state": "West", "risk_label": "Very High",
            "risk_score": 3, "trend_slope": 0.8, "level_mbgl": 22.0,
            "predicted_decline_m_per_year": 0.75,
        },
    ])
    levels = pd.DataFrame([
        {
            "district": "Alpha", "state": "North", "date": pd.Timestamp("2024-05-01"),
            "season": "pre-monsoon", "level_mbgl": 18.2, "n_stations": 3,
        },
        {
            "district": "Alpha", "state": "North", "date": pd.Timestamp("2024-12-01"),
            "season": "post-monsoon", "level_mbgl": 17.9, "n_stations": 2,
        },
        {
            "district": "Springfield", "state": "East", "date": pd.Timestamp("2024-05-01"),
            "season": "pre-monsoon", "level_mbgl": 9.0, "n_stations": 1,
        },
        {
            "district": "Springfield", "state": "West", "date": pd.Timestamp("2024-05-01"),
            "season": "pre-monsoon", "level_mbgl": 11.0, "n_stations": 1,
        },
    ])
    monkeypatch.setattr(api_module, "_public_frames", lambda: (summary, levels))
    monkeypatch.setattr(
        api_module, "_read_manifest",
        lambda: {"source_used": "synthetic", "generated_at": "2026-01-01T00:00:00+00:00"},
    )
    return TestClient(api_module.app)


def test_health_and_risk_summary(client):
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert health.json()["districts"] == 4

    risk = client.get("/api/v1/summary/risk").json()
    assert risk["total_districts"] == 4
    assert risk["counts"] == {"Low": 1, "Moderate": 1, "High": 1, "Very High": 1}


def test_district_filters_pagination_and_bounds(client):
    response = client.get("/api/v1/districts", params={"state": "north", "limit": 1})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["district"] == "Alpha"

    assert client.get("/api/v1/districts", params={"limit": 501}).status_code == 422
    assert client.get("/api/v1/districts", params={"risk": "Critical"}).status_code == 422
    all_districts = client.get("/api/v1/districts").json()["items"]
    east_springfield = next(item for item in all_districts if item["state"] == "East")
    assert east_springfield["level_mbgl"] is None


def test_district_series_not_found_and_ambiguous_names(client):
    series = client.get("/api/v1/districts/alpha/series")
    assert series.status_code == 200
    assert series.json()["items"][0]["date"] == "2024-05-01T00:00:00"

    assert client.get("/api/v1/districts/Unknown/series").status_code == 404
    ambiguous = client.get("/api/v1/districts/Springfield/series")
    assert ambiguous.status_code == 409
    assert client.get(
        "/api/v1/districts/Springfield/series", params={"state": "West"}
    ).status_code == 200


def test_api_is_read_only(client):
    assert client.post("/api/v1/districts").status_code == 405


def test_published_artifact_reader_round_trips_parquet(tmp_path, monkeypatch):
    frame = pd.DataFrame({"district": ["Alpha"], "level_mbgl": [12.5]})
    frame.to_parquet(tmp_path / "district_levels.parquet", index=False)
    monkeypatch.setattr(
        api_module, "_artifact_path",
        lambda name: tmp_path / name,
    )

    loaded = api_module._read_frame("district_levels.parquet")

    assert loaded.to_dict(orient="records") == frame.to_dict(orient="records")
