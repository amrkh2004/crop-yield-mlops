from unittest.mock import MagicMock, patch

from prodml.data import generate_synthetic_crop_data
from prodml.wandb_tracker import evaluate_wandb_model, run_wandb_experiments


def test_evaluate_wandb_model():
    """
    Tests evaluation metric calculation function for W&B logging.
    """
    X, y = generate_synthetic_crop_data(n_samples=50, random_state=42)
    mock_model = MagicMock()
    mock_model.predict.return_value = y.values

    metrics = evaluate_wandb_model(mock_model, X, y)
    assert "MAE_hg_ha" in metrics
    assert "MAE_tpha" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE_hg_ha"] == 0.0
    assert metrics["R2"] == 1.0


@patch("prodml.wandb_tracker.wandb")
def test_run_wandb_experiments_mocked(mock_wandb):
    """
    Tests W&B tracking run execution with mocked W&B API.
    """
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    results = run_wandb_experiments(mode="offline")
    assert len(results) == 3
    assert results[0]["run_name"] == "Ridge_Baseline"
    assert results[1]["run_name"] == "Random_Forest_Tuned"
    assert results[2]["run_name"] == "Gradient_Boosting"
    assert mock_wandb.log.called
    assert mock_wandb.finish.called
