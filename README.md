# 🌾 `prodml` - Crop Yield Prediction ML API

[![Python Package](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Code Coverage](https://img.shields.io/badge/Coverage-100%25-success.svg)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage-2496ED.svg)](https://www.docker.com/)

An enterprise-ready, production-grade Machine Learning REST API for predicting crop yield outputs based on environmental, agricultural, and soil input metrics.

---

## 📌 Model Overview

The `prodml` model predicts expected crop yield (measured in **total metric tons** and **tons per hectare**) using agricultural and weather inputs:
- **Crop Type** (`crop_type`): e.g., `wheat`, `rice`, `maize`, `barley`, `cotton`.
- **Cultivated Area** (`area_hectares`): Size of farming land in hectares.
- **Precipitation** (`rainfall_mm`): Seasonal rainfall in millimeters.
- **Temperature** (`temperature_celsius`): Average seasonal temperature in Celsius.
- **Fertilizer Input** (`fertilizer_kg`): Total fertilizer applied in kilograms.

---

## 🏗️ Architecture & Flow Diagram

```
                 +-------------------------------------------------+
                 |                HTTP Client / App                |
                 +-------------------------------------------------+
                                          |
                                          v
                 +-------------------------------------------------+
                 |    FastAPI REST API (async def / uvicorn)      |
                 +-------------------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
        +-------------------------------+   +-------------------------------+
        |  Correlation ID & Latency     |   |    Pydantic Schema            |
        |  Middleware (structlog JSON)  |   |    Input Validation (422)     |
        +-------------------------------+   +-------------------------------+
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                 +-------------------------------------------------+
                 |   CropYieldModel Scikit-learn Pipeline          |
                 |   (Loaded ONCE at server startup via Lifespan)  |
                 +-------------------------------------------------+
                                          |
                                          v
                 +-------------------------------------------------+
                 |  Prediction Output: yield (tons) & (tons/ha)    |
                 +-------------------------------------------------+
```

---

## ⚡ Quickstart (3 Commands)

Run the entire service locally in 3 quick commands:

### 1. Install editable Python package
```bash
pip install -e .[dev]
```

### 2. Run test suite & verify code coverage ($\ge 80\%$)
```bash
pytest --cov=src --cov-report=term-missing
```

### 3. Launch containerized API service with Docker Compose
```bash
docker compose up -d
```

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service readiness probe & model status verification |
| `POST` | `/predict` | Crop yield prediction given agricultural parameters |
| `POST` | `/feedback` | Submit actual yield observations & model corrections |

### Sample `/predict` Payload
```json
{
  "crop_type": "wheat",
  "area_hectares": 12.5,
  "rainfall_mm": 650.0,
  "temperature_celsius": 24.5,
  "fertilizer_kg": 180.0
}
```

### Sample `/predict` Response (HTTP 200 OK)
```json
{
  "request_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "crop_type": "wheat",
  "predicted_yield_tons": 33.75,
  "yield_per_hectare": 2.7,
  "status": "success"
}
```

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
  "latency_ms": 14.25,
  "timestamp": "2026-09-17T21:10:00.000Z",
  "level": "info"
}
```
