import os
import joblib
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from prodml.data import generate_synthetic_crop_data, FEATURE_NAMES
from prodml.features import build_feature_preprocessor

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import (
        StringTensorType,
        FloatTensorType,
        Int64TensorType,
    )
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False


def train_model_pipeline(
    n_samples: int = 300, random_state: int = 42
) -> Pipeline:
    """
    Trains a RandomForestRegressor pipeline on crop yield features.
    """
    X, y = generate_synthetic_crop_data(n_samples=n_samples, random_state=random_state)
    preprocessor = build_feature_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=50, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def export_model_onnx(
    pipeline: Pipeline, output_onnx_path: str
) -> bool:
    """
    Exports a trained scikit-learn pipeline to ONNX format.
    """
    if not HAS_SKL2ONNX:
        return False

    initial_types = [
        ("Area", StringTensorType([None, 1])),
        ("Item", StringTensorType([None, 1])),
        ("Year", FloatTensorType([None, 1])),
        ("average_rain_fall_mm_per_year", FloatTensorType([None, 1])),
        ("pesticides_tonnes", FloatTensorType([None, 1])),
        ("avg_temp", FloatTensorType([None, 1])),
    ]

    try:
        onnx_model = convert_sklearn(
            pipeline,
            target_opset=15,
            initial_types=initial_types,
        )
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
        with open(output_onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return True
    except Exception as e:
        print(f"ONNX conversion warning: {e}")
        return False


def save_artifacts(
    pipeline: Pipeline,
    pkl_path: str = "models/model.pkl",
    onnx_path: str = "models/model.onnx",
) -> Tuple[str, str]:
    """
    Saves trained pipeline to pickle format and ONNX format.
    """
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    joblib.dump(pipeline, pkl_path)

    onnx_success = export_model_onnx(pipeline, onnx_path)
    return pkl_path, onnx_path if onnx_success else ""


if __name__ == "__main__":
    print("Training Crop Yield Prediction Model...")
    pipe = train_model_pipeline()
    pkl_file, onnx_file = save_artifacts(pipe)
    print(f"Pickle model saved to: {pkl_file}")
    if onnx_file:
        print(f"ONNX model saved to: {onnx_file}")
    else:
        print("ONNX model export skipped/failed.")
