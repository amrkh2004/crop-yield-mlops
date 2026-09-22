import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from prodml.model import CropYieldModel


def test_model_predict_with_magic_mock():
    """
    Tests model prediction behavior by using MagicMock to isolate heavy sklearn model pipeline.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = np.array([36613.0])

    model = CropYieldModel(model_path="dummy_path.pkl")
    model.pipeline = mock_pipeline

    input_data = {
        "Area": "Albania",
        "Item": "Maize",
        "Year": 2013,
        "average_rain_fall_mm_per_year": 1485.0,
        "pesticides_tonnes": 121.0,
        "avg_temp": 16.37,
    }

    result = model.predict(input_data)

    mock_pipeline.predict.assert_called_once()
    assert result["predicted_yield_hg_ha"] == 36613.0
    assert result["predicted_yield_tons_ha"] == 3.66


def test_model_predict_raises_when_not_loaded():
    """
    Verifies RuntimeError is raised when predicting before model load.
    """
    model = CropYieldModel(model_path="non_existent.pkl")
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        model.predict(
            {
                "Area": "Egypt",
                "Item": "Wheat",
                "Year": 2024,
                "average_rain_fall_mm_per_year": 200.0,
                "pesticides_tonnes": 50.0,
                "avg_temp": 25.0,
            }
        )


def test_model_load_or_create_existing(tmp_path):
    """
    Verifies load_or_create loads existing model file using joblib.load.
    """
    dummy_model_file = tmp_path / "model.pkl"
    dummy_model_file.write_text("dummy model content")

    mock_pipeline = MagicMock()

    with (
        patch("os.path.exists", return_value=True),
        patch("joblib.load", return_value=mock_pipeline),
    ):
        model = CropYieldModel(model_path=str(dummy_model_file))
        model.load_or_create()
        assert model.pipeline == mock_pipeline


def test_model_load_or_create_fallback_baseline(tmp_path):
    """
    Verifies load_or_create trains baseline model when model file does not exist.
    """
    model_path = str(tmp_path / "new_models" / "model.pkl")
    model = CropYieldModel(model_path=model_path)
    model.load_or_create()

    assert model.pipeline is not None
    assert os.path.exists(model_path)

    prediction = model.predict(
        {
            "Area": "Albania",
            "Item": "Maize",
            "Year": 2013,
            "average_rain_fall_mm_per_year": 1485.0,
            "pesticides_tonnes": 121.0,
            "avg_temp": 16.37,
        }
    )
    assert "predicted_yield_hg_ha" in prediction
    assert prediction["predicted_yield_hg_ha"] >= 0.0
