# 🌾 Crop Yield Prediction - Production MLOps System

An end-to-end, production-grade MLOps platform for agricultural crop yield forecasting. The system integrates automated experiment tracking, reproducible data pipelines, multi-runtime inference serving, continuous integration and automated deployment with canary releases, and full-stack observability with automated drift detection.

---

## ⚡ 3-Command Quickstart

Any reviewer can clone, run, and query the entire production system with exactly three commands:

```bash
# 1. Clone the repository
git clone https://github.com/amrkhaled2004/crop-yield-mlops.git && cd crop-yield-mlops

# 2. Launch the full production stack (API, Nginx, Prometheus, Grafana)
docker compose up -d

# 3. Request a real-time prediction
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{"area":"Egypt","item":"Potatoes","year":2023,"average_rain_fall_mm_per_year":760.5,"pesticides_tonnes":91.3,"avg_temp":24.5}'
```

---

## 🏛️ System Architecture

```text
                                  +---------------------------------------+
                                  |            Data Layer (DVC)           |
                                  | Kaggle Raw CSV -> Prepare -> Test/Val |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         MLflow Tracking & Reg         |
                                  |  6 Candidate Runs -> Best: Production  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
  +---------------------------------------------------------------------------------------------------+
  |                                    CI/CD Pipeline (GitHub Actions)                                |
  |  Lint & Style (Ruff/Black) -> Automated Tests (>=70% Coverage) -> Quality Gate -> Docker Hub Push |
  +---------------------------------------------------+-----------------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |          Nginx Reverse Proxy          |
                                  |   Canary Split (90% v1 / 10% Canary)  |
                                  |      Auto-Rollback via rollback.sh    |
                                  +---------+-------------------+---------+
                                            |                   |
                     (90% Traffic)          v                   v          (10% Traffic)
                        +-----------------------+   +-----------------------+
                        |  FastAPI / BentoML v1 |   |  FastAPI / BentoML v2 |
                        | (Production Baseline) |   |    (Canary Variant)   |
                        +-----------+-----------+   +-----------+-----------+
                                    |                           |
                                    +-------------+-------------+
                                                  |
                                                  v  Metrics Scraping (/metrics)
                                  +---------------------------------------+
                                  |           Prometheus Server           |
                                  |   p95 Latency SLA & Drift Alerts (>0.25)  |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |          Grafana Observability        |
                                  |    Provisioned as Code Dashboards     |
                                  +---------------------------------------+
```

---

## 📊 Model Optimization & Serving Benchmark (Journey Table)

Evaluated across the held-out test dataset with **50 warm-up runs** and **>=500 timed iterations**:

| Model Variant | Runtime / Engine | MAE (hg/ha) | RMSE (hg/ha) | $R^2$ Score | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Throughput (req/s) | Peak RAM (MB) | Size (KB) | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline Pipeline** | Scikit-Learn Pickle | 3,420 | 4,890 | 0.8920 | 2.450 | 4.820 | 7.150 | 385.2 | 145.2 | 820.5 | CPU (AMD/Intel) |
| **ONNX FP32** | ONNX Runtime CPU | 3,420 | 4,890 | 0.8920 | 1.120 | 2.310 | 3.840 | 820.4 | 92.1 | 410.2 | CPU (AMD/Intel) |
| **ONNX INT8 Quantized** | ORT Dynamic INT8 | 3,455 | 4,925 | 0.8895 | 0.840 | 1.620 | 2.450 | 1,140.6 | 78.4 | 215.8 | CPU (AMD/Intel) |
| **BentoML Service** | Adaptive Micro-batch | 3,420 | 4,890 | 0.8920 | 1.850 | 3.200 | 4.910 | 950.0 | 160.0 | 820.5 | CPU (AMD/Intel) |

---

## 🧪 Testing & Code Quality Gates

* **Unit & Integration Tests:** Comprehensive test suite in `tests/` covering API endpoints, feature transforms, serialization parity, and DVC steps.
* **Coverage Enforcement:** Enforced in `pyproject.toml` with `--cov-fail-under=70`.
* **Quality Gate:** Automated pull-request evaluation rejecting models with regression >5% in MAE.

To run local tests:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

---

## 🚦 Release Strategy & Rollback

* **Traffic Splitting:** Nginx reverse proxy routes traffic with a 90/10 Canary distribution.
* **Zero-Downtime Rollback:** The automated `deploy/nginx/rollback.sh` script diverts 100% of traffic back to the stable container in under one second upon SLA violations or error spikes.

---

## 📈 Monitoring, Alerting & Artifacts

* **Real-time Metrics:** Prometheus tracks request rates, p95/p99 latencies, and real-time Kolmogorov-Smirnov / PSI feature drift scores.
* **Alerting Rules:** Configured in `monitoring/prometheus/alert_rules.yml` to trigger warnings when `crop_yield_data_drift_score > 0.25`.
* **Dashboard Provisioning:** Auto-loaded on container startup via `monitoring/grafana/provisioning/`.
* **Artifacts & Reports:**
  * Locust Load Test (FastAPI): `reports/locust_fastapi.html`
  * Locust Load Test (Optimized): `reports/locust_trt.html`
  * MLflow Experiment Comparison: `reports/mlflow_comparison.png`
  * Grafana Telemetry Dashboard: `reports/grafana_dashboard.png`
