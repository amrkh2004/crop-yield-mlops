# 🌾 `prodml` - Crop Yield Prediction ML API

[![Python Package](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-blue.svg)](https://onnxruntime.ai/)
[![Code Coverage](https://img.shields.io/badge/Coverage-93%25-success.svg)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)

An enterprise-ready, production-grade Machine Learning REST API for predicting crop yield outputs based on environmental, agricultural, and weather metrics.

---

## 📌 Model & Feature Overview

The `prodml` model predicts expected crop yield (measured in **hectograms per hectare `hg/ha`** and **metric tons per hectare `tons/ha`**) using:
- **Area**: Country or geographical region (e.g., `Egypt`, `Albania`, `India`, `United States of America`).
- **Item**: Crop type item (e.g., `Wheat`, `Maize`, `Potatoes`, `Rice, paddy`).
- **Year**: Harvest year (e.g., `2023`).
- **Average Rain Fall**: Average annual rainfall in millimeters (`average_rain_fall_mm_per_year`).
- **Pesticides**: Total pesticides usage in tonnes (`pesticides_tonnes`).
- **Average Temperature**: Average annual temperature in Celsius (`avg_temp`).

---

## ⚡ Quickstart (3 Commands)

Run the entire service locally in 3 quick commands:

### 1. Clone & Setup Repository
```bash
git clone https://github.com/amrkh2004/crop-yield-mlops.git
cd crop-yield-mlops
```

### 2. Launch Containerized Service with Docker Compose
```bash
docker-compose up -d --build
```

### 3. Test Prediction Endpoint (Pickle or ONNX)
```bash
curl -X POST "http://localhost:8000/predict?backend=onnx" \
     -H "Content-Type: application/json" \
     -d '{
       "area": "Egypt",
       "item": "Wheat",
       "year": 2023,
       "average_rain_fall_mm_per_year": 1200.0,
       "pesticides_tonnes": 150.0,
       "avg_temp": 24.5
     }'
```

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description | Query Params / Body |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service readiness probe & model status | None |
| `GET` | `/metadata` | Model metadata, features, and supported backends | None |
| `POST` | `/predict` | Single-item yield prediction | `?backend=pickle` or `?backend=onnx` |
| `POST` | `/predict/batch` | Batch yield prediction for multiple items | `?backend=onnx` |
| `POST` | `/feedback` | Submit actual yield observations & model feedback | `FeedbackInput` |

---

## 📋 Module 1 Deliverables Checklist Status

| Required Deliverable | Status |
| :--- | :--- |
| GitHub Repository Setup | ✅ Completed |
| `module-1-packaging` Branch | ✅ Completed |
| ML Notebook inside `notebooks/` | ✅ Completed |
| Project Structure (`src/prodml`, `tests`, `reports`, `models`) | ✅ Completed |
| `config.py` Pydantic Settings | ✅ Completed |
| Python Package / `pyproject.toml` | ✅ Completed |
| Data Module (`data.py`) | ✅ Completed |
| Features Module (`features.py`) | ✅ Completed |
| Training / Export Code (`train.py`) | ✅ Completed |
| Structured JSON Logging (`logging.py` & `middleware.py`) | ✅ Completed |
| Pickle + ONNX Serialization | ✅ Completed |
| ONNX Parity Test (`test_onnx_parity.py`) | ✅ Completed |
| Latency Comparison Benchmark | ✅ Completed |
| FastAPI Application Setup | ✅ Completed |
| Endpoint `/health` | ✅ Completed |
| Endpoint `/metadata` | ✅ Completed |
| Endpoint `/predict` | ✅ Completed |
| Endpoint `/predict/batch` | ✅ Completed |
| Pytest Suite (17 tests, 93% coverage) | ✅ Completed |
| Docker Multi-Stage Build | ✅ Completed |
| Docker Compose Setup | ✅ Completed |
| Docker Hub Readiness | ✅ Completed |
| README 3-Command Guide | ✅ Completed |
| Deliverables Report (`reports/module-1.md`) | ✅ Completed |
| PR + Review + Merge Setup | ✅ Completed |
| `v0.1.0` Release Tag | ✅ Completed |

---

## 📊 Structured JSON Logging

Logs are formatted in structured JSON via `structlog`. Every HTTP request includes correlation tracing:
```json
{
  "event": "request_processed",
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "endpoint": "/predict",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 0.38,
  "timestamp": "2026-09-19T04:30:00.000Z",
  "level": "info"
}
```
