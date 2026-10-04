# 🌾 Crop Yield Prediction - End-to-End Production MLOps System [![CI/CD Pipeline](https://github.com/amrkh2004/crop-yield-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/amrkh2004/crop-yield-mlops/actions/workflows/ci-cd.yml)

An enterprise-grade, production-ready MLOps platform for agricultural crop yield forecasting. The architecture implements reproducible data engineering pipelines, systematic experiment tracking, automated continuous integration with quality gates, multi-runtime inference serving, resilient canary releases with instant rollback, and full-stack observability with automated data drift alerting.

---

## ⚡ 3-Command Quickstart

Any reviewer can clone, run, and query the entire production system with exactly three commands:

```bash
# 1. Clone the repository
git clone https://github.com/amrkh2004/crop-yield-mlops.git && cd crop-yield-mlops

# 2. Launch the full production stack (API, Nginx, Prometheus, Grafana, MinIO)
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
| **Prediction API** | `http://localhost:8000` | Interactive OpenAPI docs at `/docs` |
| **Nginx Proxy** | `http://localhost:80` | Production Entrypoint with Canary traffic split |
| **Grafana Dashboard** | `http://localhost:3000` | User: `admin` \| Pass: `admin` |
| **Prometheus Server** | `http://localhost:9090` | Metrics target & data drift alerts |
| **MinIO Console UI** | `http://localhost:9001` | User: `minioadmin` \| Pass: `minioadmin` |
| **MinIO S3 Storage** | `http://localhost:9000` | DVC Remote storage bucket `myminio/dvcstore` |
| **MLflow UI** | `http://localhost:5000` | Model experiments & registry (`python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`) |

---

## 🏛️ System Architecture

```text
                                  +---------------------------------------+
                                  |      Data & Storage Layer (DVC)       |
                                  | Kaggle Raw CSV -> Prepare -> MinIO S3 |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         MLflow Tracking & Reg         |
                                  |  8 Candidate Runs -> Best: Production  |
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

## 📊 Model Optimization & Serving Benchmark

Evaluated on the fixed held-out test dataset with **50 warm-up runs** and **500 timed iterations**:

| Model Variant | Runtime / Engine | MAE (hg/ha) | RMSE (hg/ha) | $R^2$ Score | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Throughput (req/s) | Peak RAM (MB) | Artifact Size |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline Model** | Scikit-Learn Pickle | 5,119 | 10,609 | 0.9848 | 36.67 | 44.81 | 49.30 | 27.2 | 407.4 | 52,373.4 KB (~51.1 MB) |
| **ONNX FP32** | ONNX Runtime CPU | 5,119 | 10,609 | 0.9848 | 1.16 | 2.15 | 2.77 | 728.8 | 528.5 | 28,322.3 KB (~27.6 MB) |
| **ONNX INT8 Quantized** | ORT Dynamic INT8 | 5,119 | 10,609 | 0.9848 | 1.15 | 1.84 | 2.38 | 812.3 | 655.2 | 28,322.7 KB (~27.6 MB) |

> **Key finding:** ONNX Runtime FP32 & INT8 dynamic quantization deliver high-throughput CPU acceleration (~800 req/s vs ~27 req/s baseline) while preserving 100% regression fidelity ($R^2 = 0.9848$, $\text{MAE} = 5119$).

---

## 🧪 Testing, Quality Gates & CI/CD

* **Automated Test Suite:** Comprehensive pytest fixtures in `tests/` validating schemas, transforms, edge cases, and ONNX parity (`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`).
* **Coverage Gate:** Strictly enforced at $\ge 70\%$ in `pyproject.toml` (`--cov-fail-under=70`).
* **CI Quality Gate:** GitHub Actions blocks any commit failing tests or code coverage thresholds before container publishing.
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
  * Locust Report (FastAPI): `reports/locust_stats.csv` & `reports/locust_summary.html`
  * Benchmark Results: `reports/benchmark_results.csv` & `reports/benchmark_results.json`
  * MLflow Experiments: `src/prodml/mlflow_tracker.py`

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
├── reports/                   # Benchmark CSV/JSON reports, Locust HTML summaries
├── src/
│   ├── prodml/                # Installable Python package (core data, train, model, tracking)
│   ├── bento_service.py       # BentoML micro-batched service runner
│   └── benchmark_optimization.py # Automated benchmarking harness
├── tests/                     # Pytest suite with enforced coverage gate
├── docker-compose.yml         # Production stack orchestration (API, Canary, Nginx, Prometheus, Grafana, MinIO)
├── Dockerfile                 # Multi-stage container definition
├── pyproject.toml             # Build configuration and dependencies
└── README.md                  # System documentation and runbook
```
