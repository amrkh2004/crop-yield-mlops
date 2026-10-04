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
   - Single-request inference latency is comparable between the Baseline Pickle pipeline (p95 = 47.3 ms) and ONNX variants (FP32 p95 = 48.1 ms, INT8 p95 = 44.9 ms), as per-request Pandas feature preprocessing dominates Python execution overhead.

2. **Model Footprint & Storage Compression**:
   - INT8 Dynamic Quantization provides a massive **32.8x storage reduction**, compressing model artifact size from **46,269.7 KB** down to **1,407.1 KB** (~1.4 MB).
   - ONNX FP32 reduces baseline model disk size by ~46% down to **25,017.8 KB**.

3. **Memory Footprint (Process RSS)**:
   - Peak RAM (~354 MB) reflects overall Python process memory footprint during execution and remains consistent across variant evaluations.

4. **Fidelity & Regression Performance**:
   - As Crop Yield Prediction is a regression task, accuracy is evaluated using standard regression metrics (MAE, RMSE, R²).
   - All variants preserve exact baseline regression performance without degradation (MAE = 5119, RMSE = 10609, R² = 0.9848).

---

## 🛠️ Reproduction Command

Run the automated benchmark harness to reproduce these results:
```bash
python src/benchmark_optimization.py
```
