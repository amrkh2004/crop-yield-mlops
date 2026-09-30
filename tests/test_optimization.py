import os

from src.benchmark_optimization import prepare_onnx_models, run_benchmark_harness

from prodml.model import CropYieldModel


def test_prepare_onnx_models_and_quantization(tmp_path):
    base_model = CropYieldModel()
    base_model.load_or_create()

    fp32_path, int8_path = prepare_onnx_models(base_model)

    assert os.path.exists(fp32_path)
    assert os.path.exists(int8_path)
    assert os.path.getsize(fp32_path) > 0
    assert os.path.getsize(int8_path) > 0


def test_run_benchmark_harness_output_report(tmp_path):
    report_file = str(tmp_path / "optimization_results.json")
    report = run_benchmark_harness(output_report=report_file)

    assert os.path.exists(report_file)
    assert "variants" in report
    assert len(report["variants"]) == 3
    assert report["variants"][0]["throughput_req_sec"] > 0
