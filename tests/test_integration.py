import pytest
from fastapi.testclient import TestClient

from prodml.api.app import app

client = TestClient(app)


# --- 1. Health & Metadata Integration Tests ---

def test_integration_health_endpoint_contract():
    """
    Integration Test: /health endpoint contract, HTTP status 200, and model readiness state.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "version" in data
    assert isinstance(data["model_loaded"], bool)
    assert data["model_loaded"] is True


def test_integration_metadata_endpoint_contract():
    """
    Integration Test: /metadata endpoint contract, features list, and backend support.
    """
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()

    assert data["app_name"] == "Crop Yield Prediction API"
    assert "Area" in data["feature_names"]
    assert "Area_Item" in data["feature_names"]
    assert "pickle" in data["supported_backends"]
    assert "onnx" in data["supported_backends"]


# --- 2. /predict & /predict/batch Success & Schema Contract Tests ---

@pytest.mark.parametrize("backend", ["pickle", "onnx"])
def test_integration_predict_single_item_contract(backend):
    """
    Integration Test: /predict single item prediction schema contract for both pickle and onnx backends.
    """
    payload = {
        "area": "Egypt",
        "item": "Wheat",
        "year": 2023,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }

    response = client.post(f"/predict?backend={backend}", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "request_id" in data
    assert data["area"] == "Egypt"
    assert data["item"] == "Wheat"
    assert isinstance(data["predicted_yield_hg_ha"], float)
    assert isinstance(data["predicted_yield_tons_ha"], float)
    assert data["predicted_yield_hg_ha"] >= 0.0
    assert data["predicted_yield_tons_ha"] >= 0.0
    assert data["status"] == "success"


def test_integration_predict_batch_contract():
    """
    Integration Test: /predict/batch endpoint schema contract for processing multiple items simultaneously.
    """
    batch_payload = {
        "inputs": [
            {
                "area": "Egypt",
                "item": "Wheat",
                "year": 2023,
                "average_rain_fall_mm_per_year": 1200.0,
                "pesticides_tonnes": 150.0,
                "avg_temp": 24.5,
            },
            {
                "area": "Albania",
                "item": "Maize",
                "year": 2020,
                "average_rain_fall_mm_per_year": 1485.0,
                "pesticides_tonnes": 121.0,
                "avg_temp": 16.37,
            },
        ]
    }

    response = client.post("/predict/batch?backend=pickle", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["total_items"] == 2
    assert len(data["predictions"]) == 2

    for pred in data["predictions"]:
        assert "request_id" in pred
        assert pred["predicted_yield_hg_ha"] >= 0.0
        assert pred["predicted_yield_tons_ha"] >= 0.0


# --- 3. HTTP 422 Unprocessable Entity Validation Error Tests ---

def test_integration_predict_422_missing_required_fields():
    """
    Integration Test: /predict returns HTTP 422 when required fields are missing.
    """
    incomplete_payload = {
        "area": "Egypt",
        "item": "Wheat",
        # 'year' missing
        "average_rain_fall_mm_per_year": 1200.0,
    }

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_integration_predict_422_invalid_year_bounds():
    """
    Integration Test: /predict returns HTTP 422 when harvest year is out of range (< 1900 or > 2100).
    """
    payload_year_low = {
        "area": "Egypt",
        "item": "Wheat",
        "year": 1850,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }
    response_low = client.post("/predict", json=payload_year_low)
    assert response_low.status_code == 422

    payload_year_high = dict(payload_year_low, year=2150)
    response_high = client.post("/predict", json=payload_year_high)
    assert response_high.status_code == 422


def test_integration_predict_422_negative_values():
    """
    Integration Test: /predict returns HTTP 422 when rainfall or pesticides are negative.
    """
    payload_negative_rain = {
        "area": "Egypt",
        "item": "Wheat",
        "year": 2023,
        "average_rain_fall_mm_per_year": -50.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }
    response_rain = client.post("/predict", json=payload_negative_rain)
    assert response_rain.status_code == 422

    payload_negative_pest = dict(payload_negative_rain, average_rain_fall_mm_per_year=1200.0, pesticides_tonnes=-10.0)
    response_pest = client.post("/predict", json=payload_negative_pest)
    assert response_pest.status_code == 422


def test_integration_predict_422_out_of_range_temp():
    """
    Integration Test: /predict returns HTTP 422 when temperature is out of bounds (< -50C or > 60C).
    """
    payload_temp_extreme = {
        "area": "Egypt",
        "item": "Wheat",
        "year": 2023,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 95.0,
    }
    response = client.post("/predict", json=payload_temp_extreme)
    assert response.status_code == 422


def test_integration_predict_422_short_string_length():
    """
    Integration Test: /predict returns HTTP 422 when country or item length < 2 chars.
    """
    payload_short = {
        "area": "E",
        "item": "W",
        "year": 2023,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }
    response = client.post("/predict", json=payload_short)
    assert response.status_code == 422


def test_integration_batch_predict_422_empty_list():
    """
    Integration Test: /predict/batch returns HTTP 422 when inputs list is empty.
    """
    empty_batch = {"inputs": []}
    response = client.post("/predict/batch", json=empty_batch)
    assert response.status_code == 422


# --- 4. Feedback Endpoint Integration & 422 Validation Tests ---

def test_integration_feedback_endpoint_success_and_422():
    """
    Integration Test: /feedback endpoint records valid feedback and returns HTTP 422 for negative yield.
    """
    valid_feedback = {
        "request_id": "test-req-12345",
        "actual_yield_hg_ha": 35000.0,
        "comments": "Observed harvest yield matched model predictions closely.",
    }
    res_valid = client.post("/feedback", json=valid_feedback)
    assert res_valid.status_code == 200
    data = res_valid.json()
    assert data["status"] == "recorded"
    assert data["request_id"] == "test-req-12345"

    invalid_feedback = dict(valid_feedback, actual_yield_hg_ha=-500.0)
    res_invalid = client.post("/feedback", json=invalid_feedback)
    assert res_invalid.status_code == 422
