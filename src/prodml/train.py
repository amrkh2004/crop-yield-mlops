import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Any
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from prodml.data import generate_synthetic_crop_data, FEATURE_NAMES
from prodml.features import build_feature_preprocessor

try:
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import (
        StringTensorType,
        FloatTensorType,
        Int64TensorType,
    )
    from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxLog
    from sklearn.preprocessing import FunctionTransformer
    HAS_SKL2ONNX = True
except ImportError:
    HAS_SKL2ONNX = False


def _register_onnx_log1p_converter():
    if not HAS_SKL2ONNX:
        return
    def log1p_shape_calculator(operator):
        operator.outputs[0].type = operator.inputs[0].type

    def log1p_converter(scope, operator, container):
        op_version = container.target_opset
        x = operator.inputs[0]
        out = operator.outputs[0]
        one = np.array([1.0], dtype=np.float32)
        add_one = OnnxAdd(x, one, op_version=op_version)
        log_val = OnnxLog(add_one, op_version=op_version, output_names=[out.full_name])
        log_val.add_to(scope, container)

    try:
        update_registered_converter(
            FunctionTransformer,
            "SklearnFunctionTransformerLog1p",
            log1p_shape_calculator,
            log1p_converter,
        )
    except Exception:
        pass


_register_onnx_log1p_converter()


def train_model_pipeline(
    n_samples: int = 300, random_state: int = 42
) -> TransformedTargetRegressor:
    """
    Trains a TransformedTargetRegressor pipeline (log1p target) on crop yield features.
    """
    X, y = generate_synthetic_crop_data(n_samples=n_samples, random_state=random_state)
    preprocessor = build_feature_preprocessor(random_state=random_state)

    inner_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=inner_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X, y)
    return model


def export_model_onnx(
    model: Any, output_onnx_path: str
) -> bool:
    """
    Exports a trained scikit-learn pipeline or inner regressor to ONNX format.
    """
    if not HAS_SKL2ONNX:
        return False

    initial_types = [
        ("Area", StringTensorType([None, 1])),
        ("Item", StringTensorType([None, 1])),
        ("Area_Item", StringTensorType([None, 1])),
        ("Year", FloatTensorType([None, 1])),
        ("average_rain_fall_mm_per_year", FloatTensorType([None, 1])),
        ("pesticides_tonnes", FloatTensorType([None, 1])),
        ("avg_temp", FloatTensorType([None, 1])),
    ]

    try:
        target_pipeline = getattr(model, "regressor_", model)
        onnx_model = convert_sklearn(
            target_pipeline,
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
    pipeline: Any,
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
