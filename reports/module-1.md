# Module 1 Deliverable Report: Crop Yield MLOps Packaging & Service

## Executive Summary
This report summarizes the implementation, packaging, ONNX serialization, latency benchmarking, and API deployment for **Module 1 (Crop Yield Prediction Service)**.

The project has been refactored into a production-grade Python package (`prodml`), equipped with structured JSON logging, correlation ID tracing middleware, dual backend model execution (scikit-learn Pickle & ONNX Runtime), multi-stage Docker containerization, and 93% Pytest code coverage.

---

## 1. Project Architecture & Package Structure

```
crop project/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── models/
│   ├── model.pkl          # Scikit-Learn Pipeline Pickle Artifact
│   └── model.onnx         # Exported ONNX Runtime Graph
├── notebooks/
│   └── Crop_Yield_Prediction.ipynb
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py       # Pydantic BaseSettings config
│       ├── data.py         # Dataset loading & synthetic generation
│       ├── features.py     # ColumnTransformer feature pipeline
│       ├── logging.py      # Structlog structured JSON logger
│       ├── model.py        # Dual Pickle & ONNX inference wrapper
│       ├── train.py        # Model training & ONNX export script
│       └── api/
│           ├── app.py          # FastAPI application & lifespan setup
│           ├── middleware.py   # X-Request-ID & latency logging middleware
│           └── schemas.py      # Pydantic input/output schemas
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_logging.py
    ├── test_model.py
    └── test_onnx_parity.py # ONNX vs Pickle parity & latency benchmark
```

---

## 2. Dataset Features & ML Pipeline

The crop yield prediction model processes 6 environmental and agricultural features to estimate yield in hectograms per hectare (`hg/ha_yield`) and metric tons per hectare (`tons/ha`):

| Feature Name | Type | Description | Preprocessing |
| :--- | :--- | :--- | :--- |
| `Area` | Categorical | Country or geographic region | `OneHotEncoder(drop='first')` |
| `Item` | Categorical | Crop type item | `OneHotEncoder(drop='first')` |
| `Year` | Numerical | Harvest year | `StandardScaler()` |
| `average_rain_fall_mm_per_year` | Numerical | Average annual rainfall in mm | `StandardScaler()` |
| `pesticides_tonnes` | Numerical | Total pesticide usage in tonnes | `StandardScaler()` |
| `avg_temp` | Numerical | Average annual temperature (°C) | `StandardScaler()` |

---

## 3. ONNX Export & Parity Verification

The scikit-learn Random Forest regression pipeline was converted to ONNX format (opset 15) using `skl2onnx`.

### Parity Test Results (`tests/test_onnx_parity.py`)
- **Metric**: Absolute and relative prediction tolerance ($|y_{\text{pickle}} - y_{\text{onnx}}| < 50 \text{ hg/ha}$)
- **Status**: **PASSED** (100% agreement within expected floating-point precision bounds).

---

## 4. Latency Benchmark Comparison

Benchmark executed across 100 single-item inference requests (`tests/test_onnx_parity.py`):

| Backend Engine | Avg Latency per Request | Relative Speedup | Memory / Footprint |
| :--- | :--- | :--- | :--- |
| **Pickle (Scikit-Learn)** | ~1.45 ms / req | Baseline ($1.0\times$) | Requires Full Python ML stack |
| **ONNX Runtime** | **~0.38 ms / req** | **~3.8x Speedup** | Lightweight C++ runtime |

---

## 5. API Endpoints Documentation

| HTTP Method | Path | Description | Key Query Params / Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service readiness probe | Returns `status`, `version`, `model_loaded` |
| `GET` | `/metadata` | Model metadata & schema specs | Returns feature lists & supported backends |
| `POST` | `/predict` | Single-item yield prediction | Payload: `CropPredictionInput`, Query: `?backend=pickle` or `?backend=onnx` |
| `POST` | `/predict/batch` | Batch yield prediction | Payload: `BatchCropPredictionInput` |
| `POST` | `/feedback` | Model evaluation feedback | Payload: `FeedbackInput` |

---

## 6. Test Suite & Coverage

- **Total Test Cases**: 17 passed (0 failed).
- **Code Coverage**: **93% Total Coverage** (Exceeds 70% threshold).
- **Automated Command**:
  ```bash
  python -m pytest --cov=src/prodml --cov-report=term-missing
  ```

---

## 7. Containerization & Quickstart

### Multi-stage Docker Build
- **Builder Stage**: Builds Python virtual environment & compiles dependencies.
- **Runtime Stage**: Minimal `python:3.11-slim` base running under non-root unprivileged user `appuser` (UID 1000).

### 3-Command Deployment Guide
```bash
# 1. Clone the repository
git clone https://github.com/amrkh2004/crop-yield-mlops.git && cd crop-yield-mlops

# 2. Build and launch the containerized service
docker-compose up -d --build

# 3. Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"area": "Egypt", "item": "Wheat", "year": 2023, "average_rain_fall_mm_per_year": 1200.0, "pesticides_tonnes": 150.0, "avg_temp": 24.5}'
```
