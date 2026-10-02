# 🌾 Crop Yield Prediction - End-to-End Production MLOps System [![CI/CD Pipeline](https://github.com/amrkhaled2004/crop-yield-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amrkhaled2004/crop-yield-mlops/actions/workflows/ci-cd.yml)

An enterprise-grade, production-ready MLOps platform for agricultural crop yield forecasting. The architecture implements reproducible data engineering pipelines, systematic experiment tracking, automated continuous integration with quality gates, multi-runtime inference serving, resilient canary releases with instant rollback, and full-stack observability with automated data drift alerting.

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

## 🌐 Service Endpoints & Local Access

Once started via `docker compose up -d`, services are exposed locally:

| Service | Port / URL | Notes / Access |
| :--- | :--- | :--- |
| **Prediction API** | `http://localhost:8000` | Interactive docs at `/docs` |
| **Nginx Proxy** | `http://localhost:80` | Entrypoint with Canary split |
| **Grafana Dashboard** | `http://localhost:3000` | User: `admin` \| Pass: `admin` |
| **Prometheus Server** | `http://localhost:9090` | Metrics target & alerts |
| **MLflow UI** | `http://localhost:5000` | Model experiments & registry |

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

Evaluated on the held-out test dataset with **50 warm-up runs** and **>=500 timed iterations**:

| Model Variant | Runtime / Engine | MAE (hg/ha) | RMSE (hg/ha) | $R^2$ Score | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Throughput (req/s) | Peak RAM (MB) | Artifact Size | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline Model** | Scikit-Learn Pickle | 3,420 | 4,890 | 0.8920 | 2.450 | 4.820 | 7.150 | 385.2 | 145.2 | 820.5 KB | CPU (AMD/Intel) |
| **ONNX FP32** | ONNX Runtime CPU | 3,420 | 4,890 | 0.8920 | 1.120 | 2.310 | 3.840 | 820.4 | 92.1 | 410.2 KB | CPU (AMD/Intel) |
| **ONNX INT8 Quantized** | ORT Dynamic INT8 | 3,455 | 4,925 | 0.8895 | 0.840 | 1.620 | 2.450 | 1,140.6 | 78.4 | 215.8 KB | CPU (AMD/Intel) |
| **BentoML Service** | Adaptive Micro-batching | 3,420 | 4,890 | 0.8920 | 1.850 | 3.200 | 4.910 | 950.0 | 160.0 | 820.5 KB | CPU (AMD/Intel) |

> **Key finding:** INT8 Quantization reduced model size by 73.7% and reduced p95 latency from 4.82ms to 1.62ms with an MAE regression of only ~1%, well within the allowed SLA.

---

## 🧪 Testing, Quality Gates & CI/CD

* **Automated Test Suite:** Comprehensive pytest fixtures in `tests/` validating schemas, transforms, edge cases, and ONNX parity (`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`).
* **Coverage Gate:** Strictly enforced at $\ge 70\%$ in `pyproject.toml` (`--cov-fail-under=70`).
* **CI Quality Gate:** GitHub Actions blocks any commit if MAE regresses by more than 5%.
* **Docker Multi-Stage Build:** Non-root execution (`appuser`), slim base image, optimized layer caching.

Run tests locally:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

---

## 🚦 Release Engineering & Canary Rollback

* **Canary Traffic Split:** Nginx distributes requests with a weighted configuration (`weight=9` stable / `weight=1` canary).
* **Automated Instant Rollback:** The `deploy/nginx/rollback.sh` script restores 100% stable routing in `< 1.0` second upon detecting elevated p95 latencies or upstream errors without dropped connections.

---

## 📈 Observability, Drift Monitoring & Artifacts

* **Metrics Contract:** Service exports `/metrics` with per-stage timing, request volume counters, and prediction distribution histograms.
* **Drift Detection:** Scipy Kolmogorov-Smirnov and Total Variation Distance compute drift between training baseline and operational inference.
* **Alerting Rules:** Configured in `monitoring/prometheus/alert_rules.yml`:
  * `HighDataDriftDetected`: Fired when `crop_yield_data_drift_score > 0.25` for 2m.
  * `HighInferenceLatencyP95`: Fired when p95 response time exceeds 500ms SLA.
* **Dashboard as Code:** Provisioned automatically on Grafana boot without UI manual steps.
* **Submission Artifacts Reference:**
  * Locust Baseline Report: `reports/locust_fastapi.html`
  * Locust Optimized Report: `reports/locust_trt.html`
  * MLflow Tracking Comparison: `reports/mlflow_comparison.png`
  * Grafana Telemetry Dashboard: `reports/grafana_dashboard.png`

---

## 📂 Project Structure

```text
crop-yield-mlops/
├── .github/workflows/         # CI/CD and Continuous Training workflows
├── deploy/nginx/              # Nginx canary upstream configuration & rollback.sh
├── data/                      # DVC tracked raw & prepared data
├── models/                    # Versioned Pickle and ONNX model artifacts
├── monitoring/
│   ├── prometheus/            # Scrape configs & alerting rules
│   └── grafana/               # Provisioned datasources and dashboards as code
├── reports/                   # Benchmark charts, Locust HTML, and UI screenshots
├── src/
│   ├── prodml/                # Installable Python package (core data, train, model)
│   ├── bento_service.py       # BentoML micro-batched service runner
│   └── benchmark_optimization.py # Automated benchmarking harness
├── tests/                     # Pytest suite with enforced coverage gate
├── docker-compose.yml         # Production stack orchestration
├── Dockerfile                 # Multi-stage container definition
├── pyproject.toml             # Build configuration and dependencies
└── README.md                  # System documentation and runbook
```
