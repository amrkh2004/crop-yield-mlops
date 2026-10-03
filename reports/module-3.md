# Module 3 Report: BentoML Serving, Locust Load Testing, Canary Rollout & Airflow Retraining

## 1. Overview
Module 3 delivers advanced inference serving patterns, load testing benchmarks under high concurrency (50 users over 60 seconds), Nginx canary deployment strategies, and automated weekly model retraining.

---

## 2. BentoML Micro-Batching Web Service (`src/bento_service.py`)
- **Service Name**: `crop_yield_service`
- **Adaptive Micro-Batching**: Enabled via `@bentoml.api(batchable=True, batch_dim=0)` to combine incoming single/batch requests dynamically into vector tensor arrays.
- **Model Fetching**: Gracefully retrieves `models:/CropYieldModel/Production` from MLflow Registry with local fallback (`models/model.pkl`).
- **Health Probe**: `/healthz` probe returning readiness status for load balancers.

---

## 3. Locust Concurrency & Load Test Benchmark (`locustfile.py`)
- **Conducted Load Test Command**: `locust -f locustfile.py --headless -u 50 -r 10 -t 3m --host http://localhost --csv reports/locust --html reports/locust_summary.html`
- **Duration**: 3 minutes (180 seconds) continuous load test with 50 concurrent users at 10 users/sec spawn rate via Nginx reverse proxy on Docker Compose stack.
- **Hardware & Stack Environment**: Intel 16-core CPU, 15.7 GB RAM executing 5 Docker Compose services (`prodml-api`, `prodml-api-canary`, `nginx`, `prometheus`, `grafana`). Traffic was dynamically distributed across `prodml-api` (90%) and `prodml-api-canary` (10%).
- **Report Artifacts**:
  - HTML Interactive Report: [locust_summary.html](locust_summary.html)
  - Raw Statistics CSV: [locust_stats.csv](locust_stats.csv)
- **Empirical Benchmark Results (14,032 Total Requests Served)**:
  - `Total Requests`: 14,032 (11,150 POST `/predict`, 2,882 GET `/health`)
  - `Throughput`: 78.31 req/s
  - `Failure Rate`: 0.00% (0 errors across 180 seconds)
  - `GET /health Median (p50)`: 5 ms (Min: 1.0 ms, p95: 28 ms)
  - `POST /predict Median (p50)`: 73 ms
  - `POST /predict p95`: 240 ms
  - `POST /predict p99`: 340 ms
- **Production Performance Takeaways**:
  - **Canary Distribution**: Nginx reverse proxy load-balanced traffic across production and canary API instances seamlessly without dropped packets or socket starvation.
  - **Threadpool Efficiency**: Offloading CPU-bound inference to Starlette worker threads maintained `/health` probe median latency at **5 ms** under heavy concurrency.
  - **SLA Compliance**: Response time p95 latency remained at **240 ms**, comfortably below the 500 ms SLA threshold.

---

## 4. Airflow Model Retraining DAG (`dags/retrain_pipeline.py`)
- **Schedule**: `@weekly` with `catchup=False`.
- **Pipeline Stage Dependencies**: `extract_data_task >> train_model_task >> evaluate_model_task >> register_model_task`.
- **Quality Gate**: Computes candidate model MAE (`MAE_hg_ha`). If candidate MAE <= 10000.0 hg/ha (1.0 t/ha), model is promoted to `Production` stage in MLflow Model Registry.

---

## 5. Nginx Canary Rollout & Rollback (`docker/canary/`)
- **Canary Proxy**: Nginx reverse proxy splitting traffic between Blue (v1) and Green (v2) BentoML services.
- **Traffic Schedule**: `95/5` -> `80/20` -> `50/50` -> `0/100`.
- **Emergency Rollback Procedure**: Instant zero-downtime traffic shift back to 100% Blue via `docker exec canary_nginx_proxy nginx -s reload`.
