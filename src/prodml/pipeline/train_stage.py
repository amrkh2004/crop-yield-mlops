import os

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from prodml.data import FEATURE_NAMES, TARGET_NAME
from prodml.features import build_feature_preprocessor
from prodml.train import save_artifacts


def run_train(
    train_csv_path: str = "data/prepared/train.csv",
    output_model_path: str = "models/model.pkl",
) -> str:
    """
    DVC Pipeline Stage 2: train
    Trains ML pipeline on data/prepared/train.csv and saves trained model artifact.
    """
    if not os.path.exists(train_csv_path):
        from prodml.pipeline.prepare import run_prepare

        run_prepare()

    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[FEATURE_NAMES]
    y_train = train_df[TARGET_NAME]

    preprocessor = build_feature_preprocessor(random_state=42)
    inner_pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
            ),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=inner_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)

    pkl_path, onnx_path = save_artifacts(model, pkl_path=output_model_path)
    print(f"[DVC TRAIN] Trained model successfully saved to: {pkl_path}")
    return pkl_path


if __name__ == "__main__":
    run_train()
