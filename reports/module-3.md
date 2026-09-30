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
- **Conducted Load Test**: 100 concurrent users (`spawn-rate=10`) sending requests against the prediction endpoint for 2 minutes.
- **Payload Schema**: Realistic crop yield feature requests (`area`, `item`, `year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`).
- **Measured Latency Results**:
  - `p50 Latency`: 12.4 ms
  - `p95 Latency`: 28.6 ms
  - `p99 Latency`: 42.1 ms
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
