"""
Prometheus Metrics Instrumentation for Crop Yield ML Service.
"""

from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Counter for total HTTP prediction requests
PREDICTION_REQUESTS_TOTAL = Counter(
    "crop_yield_prediction_requests_total",
    "Total count of crop yield prediction requests",
    ["endpoint", "backend", "status"],
)

# Histogram for prediction response latency in seconds
PREDICTION_LATENCY_SECONDS = Histogram(
    "crop_yield_prediction_latency_seconds",
    "Latency of crop yield prediction requests in seconds",
    ["endpoint", "backend"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# Histogram for predicted crop yield values (hg/ha)
PREDICTED_YIELD_HG_HA = Histogram(
    "crop_yield_predicted_value_hg_ha",
    "Distribution of predicted crop yield values in hg/ha",
    ["item"],
    buckets=[10000, 50000, 100000, 200000, 300000, 500000, 750000, 1000000],
)

# Gauge for feature data drift score
DATA_DRIFT_SCORE = Gauge(
    "crop_yield_data_drift_score",
    "Overall feature data drift score across numerical and categorical features",
)

# Gauge for data drift alarm flag (1 = drift detected, 0 = normal)
DRIFT_DETECTED = Gauge(
    "crop_yield_drift_detected",
    "Binary indicator (1 or 0) showing if data drift threshold was exceeded",
)


def get_metrics_response() -> Response:
    """
    Returns Prometheus metrics plain text response for /metrics endpoint scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
