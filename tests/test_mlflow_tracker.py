from unittest.mock import MagicMock, patch

from prodml.data import generate_synthetic_crop_data
from prodml.mlflow_tracker import evaluate_model, run_mlflow_experiments


def test_evaluate_model():
    """
    Tests evaluation metric calculation function.
    """
    X, y = generate_synthetic_crop_data(n_samples=50, random_state=42)
    mock_model = MagicMock()
    mock_model.predict.return_value = y.values

    metrics = evaluate_model(mock_model, X, y)
    assert "MAE_hg_ha" in metrics
    assert "MAE_tpha" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE_hg_ha"] == 0.0
    assert metrics["R2"] == 1.0


@patch("prodml.mlflow_tracker.mlflow")
@patch("prodml.mlflow_tracker.MlflowClient")
def test_run_mlflow_experiments_mocked(mock_client_class, mock_mlflow):
    """
    Tests MLflow tracking run execution with mocked MLflow server.
    """
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.register_model.return_value = MagicMock(version="1")

    res = run_mlflow_experiments(experiment_name="Test_Exp")
    assert "best_run_id" in res
    assert "best_model_name" in res
    assert res["registered_version"] == "1"
