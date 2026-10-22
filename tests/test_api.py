"""Unit and integration tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_data

# Create a test client to send HTTP requests to the FastAPI application.
client = TestClient(app)


# Test whether the root API endpoint works correctly.
def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    # Verify that the API documentation URL is available.
    assert "documentation" in data
    assert data["documentation"] == "/docs"


# Test the health endpoint and check the API's current status.
def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()

    # Verify the service status, district count, and available models.
    assert data["status"] in ("healthy", "degraded")
    assert data["total_districts"] > 0
    assert "SARIMA" in data["models_available"]


# Test whether the API returns a list of districts within the specified limit.
def test_list_districts():
    response = client.get("/api/v1/districts?limit=5")
    assert response.status_code == 200
    districts = response.json()

    # Check the number of districts and required fields.
    assert len(districts) <= 5
    assert len(districts) > 0
    first = districts[0]
    assert "district" in first
    assert "risk_label" in first
    assert "risk_score" in first


# Test whether filtering districts by risk label works correctly.
def test_list_districts_risk_filter():
    response = client.get("/api/v1/districts?risk_label=High")
    assert response.status_code == 200
    districts = response.json()

    # Verify that every returned district has a High risk label.
    for d in districts:
        assert d["risk_label"] == "High"


# Test retrieving detailed information about a specific district.
def test_get_district_detail():
    # Fetch the first available district from the summary dataset.
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}")
    assert response.status_code == 200
    data = response.json()

    # Check the district name and its valid risk category.
    assert data["district"] == first_district
    assert data["risk_label"] in ("Low", "Moderate", "High", "Very High")


# Test whether the district forecast endpoint returns predictions from both models.
def test_get_district_forecast():
    # Select the first available district from the dataset.
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}/forecast")
    assert response.status_code == 200
    forecasts = response.json()

    # Verify that forecasts exist and include SARIMA and XGBoost.
    assert len(forecasts) > 0
    assert any(f["model"] == "SARIMA" for f in forecasts)
    assert any(f["model"] == "XGBoost" for f in forecasts)


# Test retrieving historical groundwater-level data for a district.
def test_get_district_history():
    # Select the first available district from the dataset.
    summary = get_data()["summary"]
    first_district = summary["district"].iloc[0]

    response = client.get(f"/api/v1/districts/{first_district}/history")
    assert response.status_code == 200
    history = response.json()

    # Verify that historical records exist and contain groundwater levels.
    assert len(history) > 0
    assert "level_mbgl" in history[0]


# Test whether the API returns the top groundwater depletion hotspots.
def test_get_hotspots():
    response = client.get("/api/v1/hotspots?limit=3")
    assert response.status_code == 200
    hotspots = response.json()

    # Check that the result contains at least one hotspot and respects the limit.
    assert len(hotspots) <= 3
    assert len(hotspots) > 0


# Test whether the SHAP endpoint returns feature-importance information.
def test_get_shap_importance():
    response = client.get("/api/v1/explain/shap")
    assert response.status_code == 200
    shap_data = response.json()

    # Verify that feature names and their SHAP importance values are present.
    assert len(shap_data) > 0
    assert "feature" in shap_data[0]
    assert "mean_abs_shap" in shap_data[0]


# Test whether the API returns model evaluation metrics.
def test_get_metrics():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    metrics = response.json()

    # Check that metrics for XGBoost and SARIMA backtesting are available.
    assert "xgboost" in metrics
    assert "sarima_backtest" in metrics


# Test whether requesting a nonexistent district returns a 404 error.
def test_district_not_found():
    response = client.get("/api/v1/districts/NonExistentDistrictName9999")

    # Verify the HTTP status code and the error message.
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()