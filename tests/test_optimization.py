import os

from src.benchmark_optimization import prepare_onnx_models, run_benchmark_harness
from prodml.model import CropYieldModel


def test_prepare_onnx_models_and_quantization(tmp_path):
    pkl_file = str(tmp_path / "model.pkl")
    onnx_file = str(tmp_path / "model.onnx")
    int8_file = str(tmp_path / "model_int8.onnx")

    base_model = CropYieldModel(model_path=pkl_file, onnx_path=onnx_file)
    base_model.load_or_create()

    fp32_path, int8_path = prepare_onnx_models(base_model, fp32_path=onnx_file, int8_path=int8_file)

    assert os.path.exists(fp32_path)
    assert os.path.exists(int8_path)
    assert os.path.getsize(fp32_path) > 0
    assert os.path.getsize(int8_path) > 0


def test_run_benchmark_harness_output_report(tmp_path):
    report_file = str(tmp_path / "optimization_results.json")
    csv_file = str(tmp_path / "benchmark_results.csv")
    json_file = str(tmp_path / "benchmark_results.json")
    pkl_file = str(tmp_path / "model.pkl")
    onnx_file = str(tmp_path / "model.onnx")
    int8_file = str(tmp_path / "model_int8.onnx")

    report = run_benchmark_harness(
        output_report=report_file,
        csv_report=csv_file,
        json_report=json_file,
        warmup_runs=5,
        num_runs=10,
        model_path=pkl_file,
        onnx_path=onnx_file,
        int8_path=int8_file,
    )

    assert os.path.exists(report_file)
    assert "variants" in report
    assert len(report["variants"]) == 3
    assert report["variants"][0]["throughput_req_sec"] > 0
