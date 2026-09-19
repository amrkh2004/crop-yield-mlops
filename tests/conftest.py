import pytest
from fastapi.testclient import TestClient
from prodml.api.app import app


@pytest.fixture
def client():
    """
    TestClient fixture for making API calls without starting live server.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_prediction_payload():
    return {
        "area": "Albania",
        "item": "Maize",
        "year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 16.37,
    }


@pytest.fixture
def invalid_prediction_payload():
    return {
        "area": "A",  # min_length violation
        "item": "",  # min_length violation
        "year": 1800,  # ge=1900 violation
        "average_rain_fall_mm_per_year": -100.0,  # ge=0 violation
        "pesticides_tonnes": -50.0,  # ge=0 violation
        "avg_temp": 100.0,  # le=60.0 violation
    }


@pytest.fixture
def sample_feedback_payload():
    return {
        "request_id": "test-request-12345",
        "actual_yield_hg_ha": 36613.0,
        "comments": "Actual yield recorded from field observation.",
    }
