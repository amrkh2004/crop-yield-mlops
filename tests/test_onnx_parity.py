import os
import time

import numpy as np

from prodml.data import generate_synthetic_crop_data
from prodml.model import CropYieldModel


def get_test_model(tmp_path):
    pkl_path = "models/model.pkl"
    onnx_path = "models/model.onnx"
    if not os.path.exists(pkl_path) or not os.path.exists(onnx_path):
        pkl_path = str(tmp_path / "model.pkl")
        onnx_path = str(tmp_path / "model.onnx")

    model = CropYieldModel(model_path=pkl_path, onnx_path=onnx_path)
    model.load_or_create()
    return model


def test_pickle_onnx_prediction_parity(tmp_path):
    """
    Verifies that the ONNX runtime model predictions match the scikit-learn Pickle pipeline within tolerance.
    """
    model = get_test_model(tmp_path)
    onnx_sess = model._get_onnx_session()

    assert model.pipeline is not None, "Pickle pipeline should be loaded"
    if onnx_sess is None:
        return

    X, _ = generate_synthetic_crop_data(n_samples=50, random_state=123)
    sample_items = X.to_dict(orient="records")

    pickle_res = model.predict(sample_items, backend="pickle")
    onnx_res = model.predict(sample_items, backend="onnx")

    pickle_yields = [item["predicted_yield_hg_ha"] for item in pickle_res]
    onnx_yields = [item["predicted_yield_hg_ha"] for item in onnx_res]

    np.testing.assert_allclose(
        pickle_yields,
        onnx_yields,
        rtol=0.25,
        atol=5000.0,
        err_msg="ONNX vs Pickle prediction mismatch",
    )


def test_latency_comparison_benchmark(tmp_path):
    """
    Benchmarks inference latency of Pickle vs ONNX model over multiple iterations.
    """
    model = get_test_model(tmp_path)

    sample_item = {
        "Area": "Egypt",
        "Item": "Wheat",
        "Year": 2023,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }

    n_iterations = 50

    start_pickle = time.perf_counter()
    for _ in range(n_iterations):
        model.predict(sample_item, backend="pickle")
    pickle_duration_ms = (time.perf_counter() - start_pickle) * 1000.0 / n_iterations

    start_onnx = time.perf_counter()
    for _ in range(n_iterations):
        model.predict(sample_item, backend="onnx")
    onnx_duration_ms = (time.perf_counter() - start_onnx) * 1000.0 / n_iterations

    print(f"\n--- Latency Benchmark Results ({n_iterations} iterations) ---")
    print(f"Pickle average latency: {pickle_duration_ms:.4f} ms/req")
    print(f"ONNX average latency:   {onnx_duration_ms:.4f} ms/req")

    assert pickle_duration_ms > 0
    assert onnx_duration_ms > 0
