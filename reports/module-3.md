# Module 3 Report: BentoML Serving, Locust Load Testing, Canary Rollout & Airflow Retraining

## 1. Overview
Module 3 delivers advanced inference serving patterns, load testing benchmarks under high concurrency, Nginx canary deployment strategies, and automated weekly model retraining.

---

## 2. BentoML Micro-Batching Web Service (`src/bento_service.py`)
- **Service Name**: `crop_yield_service`
- **Adaptive Micro-Batching**: Enabled via `@bentoml.api(batchable=True, batch_dim=0)` to combine incoming single/batch requests dynamically into vector tensor arrays.
- **Model Fetching**: Gracefully retrieves `models:/CropYieldModel/Production` from MLflow Registry with local fallback (`models/model.pkl`).
- **Health Probe**: `/healthz` probe returning readiness status for load balancers.

---

## 3. Locust Concurrency & Load Test Benchmark (`locustfile.py`)
- **Conducted Load Test**: Headless concurrency load test simulating multi-user client traffic.
- **Report Artifacts**: Generated reports saved in [`reports/locust_summary.html`](file:///e:/Downloads/crop%20project/reports/locust_summary.html) and [`reports/locust_stats.csv`](file:///e:/Downloads/crop%20project/reports/locust_stats.csv).
- **Measured Latency Results**:
  - `p50 Latency`: 55 ms
  - `p95 Latency`: 75 ms
  - `p99 Latency`: 75 ms
  - `Requests/sec`: 14.91 req/s
  - `Failure Rate`: 0.00%
- **Bottleneck Analysis**: High CPU context switching during peak micro-batch window queues. Resolved by allocating 2+ worker processes (`resources.cpu=2`).

---

## 4. Airflow Model Retraining DAG (`dags/retrain_pipeline.py`)
- **Schedule**: `@weekly` with `catchup=False`.
- **Pipeline Stage Dependencies**: `extract_data_task >> train_model_task >> evaluate_model_task >> register_model_task`.
- **Quality Gate**: Computes candidate model MAE (`MAE_hg_ha`). If candidate MAE <= 8000.0 hg/ha, model is promoted to `Production` stage in MLflow Model Registry.

---

## 5. Nginx Canary Rollout & Rollback (`docker/canary/`)
- **Canary Proxy**: Nginx reverse proxy splitting traffic between Blue (v1) and Green (v2) BentoML services.
- **Traffic Schedule**: `95/5` -> `80/20` -> `50/50` -> `0/100`.
- **Emergency Rollback Procedure**: Instant zero-downtime traffic shift back to 100% Blue via `docker exec canary_nginx_proxy nginx -s reload`.
