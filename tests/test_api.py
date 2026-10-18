"""Unit and integration tests for the FastAPI service."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_data

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert data["documentation"] == "/docs"


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["total_districts"] > 0
    assert "SARIMA" in data["models_available"]


def test_list_districts():
    response = client.get("/api/v1/districts?limit=5")
    assert response.status_code == 200
    districts = response.json()
    assert len(districts) <= 5
    assert len(districts) > 0
    first = districts[0]
    assert "district" in first
    assert "risk_label" in first
    assert "risk_score" in first


def test_list_districts_risk_filter():
    response = client.get("/api/v1/districts?risk_label=High")
    assert response.status_code == 200
    districts = response.json()
    for d in districts:
        assert d["risk_label"] == "High"


def test_get_district_detail():
    # Fetch first available district
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}")
    assert response.status_code == 200
    data = response.json()
    assert data["district"] == first_district
    assert data["risk_label"] in ("Low", "Moderate", "High", "Very High")


def test_get_district_forecast():
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}/forecast")
    assert response.status_code == 200
    forecasts = response.json()
    assert len(forecasts) > 0
    assert any(f["model"] == "SARIMA" for f in forecasts)
    assert any(f["model"] == "XGBoost" for f in forecasts)


def test_get_district_history():
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}/history")
    assert response.status_code == 200
    history = response.json()
    assert len(history) > 0
    assert "level_mbgl" in history[0]


def test_get_hotspots():
    response = client.get("/api/v1/hotspots?limit=3")
    assert response.status_code == 200
    hotspots = response.json()
    assert len(hotspots) <= 3
    assert len(hotspots) > 0


def test_get_shap_importance():
    response = client.get("/api/v1/explain/shap")
    assert response.status_code == 200
    shap_data = response.json()
    assert len(shap_data) > 0
    assert "feature" in shap_data[0]
    assert "mean_abs_shap" in shap_data[0]


def test_get_metrics():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "xgboost" in metrics
    assert "sarima_backtest" in metrics


def test_district_not_found():
    response = client.get("/api/v1/districts/NonExistentDistrictName9999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
