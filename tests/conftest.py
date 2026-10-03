import os

import pytest
from fastapi.testclient import TestClient

from prodml.api.app import app


@pytest.fixture(autouse=True)
def protect_production_models(monkeypatch, tmp_path):
    """
    Protects production models in models/ directory from being overwritten during unit test execution.
    Intercepts any save_artifacts calls targeting models/ and diverts them to tmp_path.
    """
    import prodml.train

    original_save = prodml.train.save_artifacts

    def safe_save_artifacts(pipeline, pkl_path="models/model.pkl", onnx_path="models/model.onnx"):
        norm_pkl = os.path.normpath(pkl_path)
        norm_onnx = os.path.normpath(onnx_path)
        if norm_pkl == "models/model.pkl" or norm_pkl == "models\\model.pkl" or norm_pkl.startswith("models"):
            pkl_path = str(tmp_path / os.path.basename(pkl_path))
        if norm_onnx == "models/model.onnx" or norm_onnx == "models\\model.onnx" or norm_onnx.startswith("models"):
            onnx_path = str(tmp_path / os.path.basename(onnx_path))
        return original_save(pipeline, pkl_path, onnx_path)

    monkeypatch.setattr(prodml.train, "save_artifacts", safe_save_artifacts)
    try:
        monkeypatch.setattr("prodml.model.save_artifacts", safe_save_artifacts)
    except Exception:
        pass
    try:
        monkeypatch.setattr("prodml.mlflow_tracker.save_artifacts", safe_save_artifacts)
    except Exception:
        pass


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
