"""
Unit and Integration Tests for Prometheus Metrics Endpoint and Data Drift Detection.
"""

import os
import json
import pandas as pd
from fastapi.testclient import TestClient

from prodml.api.app import app
from prodml.drift_detector import DataDriftDetector

client = TestClient(app)


def test_metrics_endpoint():
    """
    Verify /metrics endpoint responds with HTTP 200 and Prometheus plain text metrics.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"] or "version=0.0.4" in response.headers["content-type"]
    assert "crop_yield_prediction_requests_total" in response.text
    assert "crop_yield_prediction_latency_seconds" in response.text


def test_prediction_telemetry():
    """
    Verify /predict endpoint updates Prometheus metrics counters.
    """
    payload = {
        "area": "Egypt",
        "item": "Maize",
        "year": 2023,
        "average_rain_fall_mm_per_year": 120.0,
        "pesticides_tonnes": 45.0,
        "avg_temp": 24.5,
    }
    response = client.post("/predict?backend=pickle", json=payload)
    assert response.status_code == 200

    metrics_resp = client.get("/metrics")
    assert response.status_code == 200
    assert 'crop_yield_prediction_requests_total{backend="pickle",endpoint="/predict",status="success"}' in metrics_resp.text


def test_data_drift_detector_normal(tmp_path):
    """
    Verify DataDriftDetector returns drift_detected=False for identical reference/current distributions.
    """
    ref_df = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        "average_rain_fall_mm_per_year": [100.0, 105.0, 110.0, 108.0, 112.0, 102.0, 106.0, 109.0, 111.0, 104.0],
        "pesticides_tonnes": [20.0, 22.0, 21.0, 23.0, 22.5, 20.5, 21.5, 22.0, 23.5, 21.0],
        "avg_temp": [20.0, 20.5, 21.0, 20.2, 20.8, 20.1, 20.6, 20.4, 20.9, 20.3],
        "Area": ["Egypt"] * 10,
        "Item": ["Maize"] * 10,
    })

    detector = DataDriftDetector(p_value_threshold=0.05)
    report_path = str(tmp_path / "test_drift_normal.json")
    result = detector.detect_drift(ref_df, ref_df.copy(), export_json_path=report_path)

    assert result["drift_detected"] is False
    assert result["drifted_features_count"] == 0
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
        assert saved_data["drift_detected"] is False


def test_data_drift_detector_shifted(tmp_path):
    """
    Verify DataDriftDetector flags drift when feature distributions are significantly shifted.
    """
    ref_df = pd.DataFrame({
        "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        "average_rain_fall_mm_per_year": [100.0, 105.0, 110.0, 108.0, 112.0, 102.0, 106.0, 109.0, 111.0, 104.0],
        "pesticides_tonnes": [20.0, 22.0, 21.0, 23.0, 22.5, 20.5, 21.5, 22.0, 23.5, 21.0],
        "avg_temp": [20.0, 20.5, 21.0, 20.2, 20.8, 20.1, 20.6, 20.4, 20.9, 20.3],
        "Area": ["Egypt"] * 10,
        "Item": ["Maize"] * 10,
    })

    shifted_df = ref_df.copy()
    shifted_df["pesticides_tonnes"] = shifted_df["pesticides_tonnes"] * 10.0
    shifted_df["avg_temp"] = shifted_df["avg_temp"] + 25.0

    detector = DataDriftDetector(p_value_threshold=0.05)
    report_path = str(tmp_path / "test_drift_shifted.json")
    result = detector.detect_drift(ref_df, shifted_df, export_json_path=report_path)

    assert result["drift_detected"] is True
    assert result["drifted_features_count"] >= 1
    assert result["retraining_recommended"] is True
