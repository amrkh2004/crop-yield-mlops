import os
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from prodml.data import CATEGORICAL_FEATURES, FEATURE_NAMES, NUMERIC_FEATURES
from prodml.train import save_artifacts, train_model_pipeline

try:
    import onnxruntime as ort

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class CropYieldModel:
    """
    ML Pipeline wrapper for Crop Yield Prediction.
    Supports both Pickle (scikit-learn) and ONNX runtime backends.
    """

    FEATURE_NAMES = FEATURE_NAMES
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
    NUMERIC_FEATURES = NUMERIC_FEATURES

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        onnx_path: str = "models/model.onnx",
        backend: str = "pickle",
    ):
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.backend = backend
        self.pipeline = None
        self.ort_session = None

    def load_or_create(self) -> None:
        """
        Loads trained model pipeline from disk (Pickle or ONNX) if available.
        Otherwise trains, exports, and loads both formats.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.onnx_path):
            pipeline = train_model_pipeline()
            save_artifacts(pipeline, self.model_path, self.onnx_path)

        # Load Pickle pipeline
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)

        # Load ONNX session if available
        if HAS_ONNXRUNTIME and os.path.exists(self.onnx_path):
            try:
                self.ort_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
            except Exception as e:
                print(f"ONNX session init warning: {e}")
                self.ort_session = None

    def predict(
        self, input_data: Union[Dict[str, Any], List[Dict[str, Any]]], backend: str = None
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Runs model prediction for a single input or a batch of inputs.
        Backend can be specified explicitly ('pickle' or 'onnx').
        """
        use_backend = backend or self.backend

        is_batch = isinstance(input_data, list)
        items = [dict(item) for item in (input_data if is_batch else [input_data])]
        for item in items:
            if "Area_Item" not in item and "Area" in item and "Item" in item:
                item["Area_Item"] = f"{item['Area']}_{item['Item']}"

        df = pd.DataFrame(items)[self.FEATURE_NAMES]

        if use_backend == "onnx" and self.ort_session is not None:
            raw_log_preds = self._predict_onnx(df)
            raw_preds = np.expm1(raw_log_preds)
        else:
            if self.pipeline is None:
                raise RuntimeError("Model is not loaded. Call load_or_create() first.")
            raw_preds = self.pipeline.predict(df)

        results = []
        for raw_pred in raw_preds:
            pred_val = float(raw_pred)
            predicted_hg_ha = max(round(pred_val, 2), 0.0)
            predicted_tons_ha = round(predicted_hg_ha / 10000.0, 2)
            results.append(
                {
                    "predicted_yield_hg_ha": predicted_hg_ha,
                    "predicted_yield_tons_ha": predicted_tons_ha,
                }
            )

        return results if is_batch else results[0]

    def _predict_onnx(self, df: pd.DataFrame) -> np.ndarray:
        """
        Internal helper for ONNX inference. Returns predictions in log scale.
        """
        inputs = {}
        for col in self.FEATURE_NAMES:
            if col in self.CATEGORICAL_FEATURES:
                inputs[col] = np.array(df[col].astype(str).tolist(), dtype=object).reshape(-1, 1)
            else:
                inputs[col] = df[col].values.astype(np.float32).reshape(-1, 1)

        output_name = self.ort_session.get_outputs()[0].name
        onnx_outputs = self.ort_session.run([output_name], inputs)
        return onnx_outputs[0].flatten()
