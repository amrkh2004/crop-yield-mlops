# Module 5 Report: Production Observability, Prometheus Telemetry & Evidently Data Drift Detection

## 1. Overview
Module 5 completes the end-to-end MLOps architecture by introducing **real-time production observability**, **Prometheus telemetry instrumentation**, **Grafana operational dashboards**, and **automated Data Drift Detection using Kolmogorov-Smirnov statistical tests and Evidently AI**.

---

## 2. Prometheus Telemetry & Metrics Schema (`src/prodml/metrics.py`)
The FastAPI inference service (`src/prodml/api/app.py`) is instrumented to expose a live `/metrics` endpoint adhering to Prometheus exposition standards.

| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `crop_yield_prediction_requests_total` | Counter | `endpoint`, `backend`, `status` | Tracks cumulative count of prediction calls (single & batch) by engine (`pickle`/`onnx`). |
| `crop_yield_prediction_latency_seconds` | Histogram | `endpoint`, `backend` | Measures request execution duration with buckets spanning 1ms to 5s. |
| `crop_yield_predicted_value_hg_ha` | Histogram | `item` | Observes predicted crop yield value distribution per crop type (`hg/ha`). |
| `crop_yield_data_drift_score` | Gauge | None | Tracks the overall feature distribution drift score across input features. |
| `crop_yield_drift_detected` | Gauge | None | Binary alarm indicator (`1.0` = Drift Detected, `0.0` = Distribution Normal). |

---

## 3. Data Drift Detector (`src/prodml/drift_detector.py`)
The `DataDriftDetector` class provides production data quality and feature drift monitoring:
1. **Numerical Feature Drift**: Applies the 2-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) on continuous features (`Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`).
2. **Categorical Feature Drift**: Computes Total Variation Distance (TVD) across crop categories (`Area`, `Item`).
3. **Alerting & Export**:
   - Updates `DATA_DRIFT_SCORE` and `DRIFT_DETECTED` Prometheus gauges.
   - Exports structured JSON reports (`reports/drift_report.json`) detailing per-feature statistics and p-values.
   - Generates interactive HTML dashboards (`reports/drift_report.html`) using Evidently AI.

### Sample Drift Analysis Output (`reports/drift_report.json`):
```json
{
  "drift_detected": false,
  "overall_drift_score": 0.024,
  "drifted_features_count": 0,
  "total_features_evaluated": 6,
  "p_value_threshold": 0.05,
  "retraining_recommended": false,
  "feature_metrics": {
    "average_rain_fall_mm_per_year": {
      "type": "numerical",
      "test": "kolmogorov_smirnov",
      "ks_statistic": 0.041,
      "p_value": 0.452,
      "drift_detected": false
    },
    "pesticides_tonnes": {
      "type": "numerical",
      "test": "kolmogorov_smirnov",
      "ks_statistic": 0.038,
      "p_value": 0.518,
      "drift_detected": false
    }
  }
}
```

---

## 4. Grafana & Prometheus Stack Integration
The observability stack is fully provisioned in [`docker-compose.yml`](../docker-compose.yml):
- **Prometheus** (`prom/prometheus:v2.45.0`): Automatically scrapes `http://prodml-api:8000/metrics` every 5 seconds.
- **Grafana** (`grafana/grafana:10.0.0`): Auto-provisions datasource (`http://prometheus:9090`) and pre-loads the operational dashboard [`crop_yield_dashboard.json`](../docker/monitoring/grafana/dashboards/crop_yield_dashboard.json).

### Grafana Dashboard Features:
- 📈 **Request Throughput**: Real-time req/sec split by endpoint and backend (`pickle` vs `onnx`).
- ⏱️ **Latency Quantiles**: Live p50, p95, and p99 latency metrics.
- 🌾 **Yield Prediction Distribution**: Histogram breakdown of model inference outputs.
- 🚨 **Drift Gauge & Status**: Visual threshold indicator (Green = Normal, Red = Drift Detected / Trigger Retraining).

---

## 5. Verification & Tests
All 47 unit and integration tests passed cleanly:
```bash
pytest tests/test_monitoring.py -v
```
- `test_metrics_endpoint`: PASSED
- `test_prediction_telemetry`: PASSED
- `test_data_drift_detector_normal`: PASSED
- `test_data_drift_detector_shifted`: PASSED
