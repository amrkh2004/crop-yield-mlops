# ⚡ Module 4: Model Optimization & Benchmark Harness

This report documents the performance, latency, memory footprint, and model size optimization journey for the **Crop Yield Prediction** ML model using **ONNX Export** and **INT8 Dynamic Quantization**.

---

## 📊 Current Benchmark Results

| Model Variant | MAE | RMSE | R² | p95 Latency (ms) | Throughput (Req/Sec) | Peak RAM (MB) | Size (KB) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Baseline Model** | 5119 | 10609 | 0.9848 | 47.277 | 25.2 | 353.89 | 46269.7 |
| **ONNX FP32** | 5119 | 10609 | 0.9848 | 48.093 | 25.4 | 353.98 | 25017.8 |
| **ONNX INT8 Quantized** | 5119 | 10609 | 0.9848 | 44.870 | 28.2 | 354.03 | 1407.1 |

---

## Benchmark Configuration

- Warmup iterations: 50
- Timed benchmark iterations: 500
- Evaluation dataset: Fixed held-out test set
- Metrics: MAE, RMSE, R²
- Latency: p50, p95, p99
- Throughput: Requests/sec
- Memory: Peak RSS
- Hardware/environment information: recorded by the benchmark harness

---

## 💡 Trade-off Analysis & Findings

1. **Inference Latency & Throughput**:
   - ONNX INT8 Quantized model achieved the best latency performance at **44.870 ms (p95)** and highest throughput at **28.2 requests/sec**.

2. **Model Footprint & Compression**:
   - INT8 Dynamic Quantization achieved drastic model size reduction down to **1,407.1 KB**, compared to **46,269.7 KB** for baseline model.

3. **Fidelity & Regression Performance**:
   - As Crop Yield Prediction is a regression task, model performance is evaluated using standard regression metrics (MAE, RMSE, R²).
   - All variants preserve full model accuracy without degradation (MAE = 5119, RMSE = 10609, R² = 0.9848).

---

## 🛠️ Reproduction Command

Run the automated benchmark harness to reproduce these results:
```bash
python src/benchmark_optimization.py
```
