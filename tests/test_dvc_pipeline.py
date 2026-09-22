import os

from prodml.pipeline.evaluate import run_evaluate
from prodml.pipeline.prepare import run_prepare
from prodml.pipeline.train_stage import run_train


def test_dvc_pipeline_end_to_end(tmp_path):
    """
    Verifies that the DVC pipeline stages (prepare -> train -> evaluate) execute cleanly end-to-end.
    """
    raw_path = str(tmp_path / "raw.csv")
    prep_dir = str(tmp_path / "prepared")
    model_path = str(tmp_path / "model.pkl")
    metrics_path = str(tmp_path / "metrics.json")

    # 1. Prepare stage
    train_path, test_path = run_prepare(raw_csv_path=raw_path, output_dir=prep_dir)
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)

    # 2. Train stage
    trained_model_path = run_train(train_csv_path=train_path, output_model_path=model_path)
    assert os.path.exists(trained_model_path)

    # 3. Evaluate stage
    metrics = run_evaluate(test_csv_path=test_path, model_path=model_path, output_metrics_path=metrics_path)
    assert os.path.exists(metrics_path)
    assert "MAE_hg_ha" in metrics
    assert "R2" in metrics
