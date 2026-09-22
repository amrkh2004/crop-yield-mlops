import json
import os
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prodml.data import FEATURE_NAMES, TARGET_NAME


def run_evaluate(
    test_csv_path: str = "data/prepared/test.csv",
    model_path: str = "models/model.pkl",
    output_metrics_path: str = "reports/metrics.json",
) -> Dict[str, float]:
    """
    DVC Pipeline Stage 3: evaluate
    Evaluates trained model on data/prepared/test.csv and exports reports/metrics.json.
    """
    if not os.path.exists(test_csv_path) or not os.path.exists(model_path):
        from prodml.pipeline.train_stage import run_train

        run_train()

    test_df = pd.read_csv(test_csv_path)
    X_test = test_df[FEATURE_NAMES]
    y_test = test_df[TARGET_NAME]

    model = joblib.load(model_path)
    preds = np.clip(model.predict(X_test), 0, None)

    mae_hg = float(mean_absolute_error(y_test, preds))
    mae_tpha = float(mae_hg / 10000.0)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "MAE_hg_ha": round(mae_hg, 2),
        "MAE_tpha": round(mae_tpha, 4),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "n_test_samples": len(test_df),
    }

    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DVC EVALUATE] Metrics report exported to: {output_metrics_path}")
    print(f"[DVC EVALUATE] MAE (hg/ha): {metrics['MAE_hg_ha']} | R2: {metrics['R2']}")
    return metrics


if __name__ == "__main__":
    run_evaluate()
