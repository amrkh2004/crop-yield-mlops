"""
Additional Unit Tests to ensure >80% test coverage across all src modules.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

try:
    from batch_score import get_production_model, run_batch_scoring
    from consumer import predict_on_event, run_consumer, store_result
    from event_producer import run_latency_benchmark
    from vllm_client import run_vllm_benchmark
except ImportError:
    from src.batch_score import get_production_model, run_batch_scoring
    from src.consumer import predict_on_event, run_consumer, store_result
    from src.event_producer import run_latency_benchmark
    from src.vllm_client import run_vllm_benchmark

from prodml.data import generate_synthetic_crop_data, load_raw_crop_data
from prodml.drift_detector import DataDriftDetector


def test_generate_synthetic_crop_data():
    """Test synthetic data generator."""
    X, y = generate_synthetic_crop_data(n_samples=50, random_state=42)
    assert len(X) == 50
    assert len(y) == 50
    assert "Area_Item" in X.columns


def test_load_raw_crop_data_nonexistent(tmp_path):
    """Test loading raw crop data fallback when file does not exist."""
    fake_path = str(tmp_path / "non_existent.csv")
    X, y = load_raw_crop_data(filepath=fake_path)
    assert len(X) == 600
    assert len(y) == 600


def test_load_raw_crop_data_real_kaggle():
    """Test loading real Kaggle crop yield dataset."""
    X, y = load_raw_crop_data()
    assert len(X) > 20000
    assert len(y) > 20000
    assert getattr(X, "attrs", {}).get("dataset_tag") == "kaggle_crop_yield_real"


def test_batch_score_end_to_end(tmp_path):
    """Test batch scoring script execution."""
    input_dir = str(tmp_path / "input")
    output_dir = str(tmp_path / "output")
    out_file = run_batch_scoring(input_dir=input_dir, output_dir=output_dir)
    assert out_file.endswith(".parquet")


def test_get_production_model_fallback():
    """Test get_production_model fallback when MLflow fails."""
    model = get_production_model("NonExistentModel")
    assert model is not None


def test_consumer_predict_and_store(tmp_path):
    """Test consumer event prediction and result storing."""
    payload = {
        "Area": "Egypt",
        "Item": "Wheat",
        "Year": 2023,
        "average_rain_fall_mm_per_year": 1200.0,
        "pesticides_tonnes": 150.0,
        "avg_temp": 24.5,
    }
    result = predict_on_event("test_evt_001", payload)
    assert "event_id" in result
    assert result["predicted_yield_hg_ha"] > 0.0

    out_dir = str(tmp_path / "events")
    store_result("test_evt_002", payload, 25000.0, 1.5, output_dir=out_dir)


@patch("redis.Redis")
def test_run_consumer_mock(mock_redis):
    """Test consumer main loop with Redis fallback."""
    run_consumer(redis_host="invalid_host_for_test", max_iterations=1)


def test_event_producer_benchmark():
    """Test event producer latency benchmark runner."""
    results = run_latency_benchmark(target_rate=50, num_events=10)
    assert "achieved_rate" in results
    assert "p50_latency_ms" in results


@patch("openai.OpenAI")
def test_vllm_client_mock(mock_openai):
    """Test vLLM benchmark client with mocked OpenAI API."""
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Word"

    mock_client_inst = MagicMock()
    mock_client_inst.chat.completions.create.return_value = [mock_chunk]
    mock_openai.return_value = mock_client_inst

    bench_res = run_vllm_benchmark()
    assert "ttft_ms" in bench_res


def test_drift_detector_html_export(tmp_path):
    """Test drift detector HTML report generation and fallback."""
    ref_df = pd.DataFrame(
        {
            "Year": [2010, 2011, 2012, 2013, 2014],
            "average_rain_fall_mm_per_year": [100.0, 105.0, 110.0, 108.0, 112.0],
            "pesticides_tonnes": [20.0, 22.0, 21.0, 23.0, 22.5],
            "avg_temp": [20.0, 20.5, 21.0, 20.2, 20.8],
            "Area": ["Egypt"] * 5,
            "Item": ["Maize"] * 5,
        }
    )

    detector = DataDriftDetector()
    html_path = str(tmp_path / "drift.html")
    detector.generate_evidently_html_report(ref_df, ref_df.copy(), export_html_path=html_path)
