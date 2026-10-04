from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from prodml.data import load_raw_crop_data
from prodml.features import build_feature_preprocessor
from prodml.logging import get_logger
from prodml.train import save_artifacts

logger = get_logger("prodml.wandb_tracker")

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def evaluate_wandb_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates predictions and computes MAE (hg/ha), MAE (t/ha), RMSE, and R2.
    """
    preds = np.clip(model.predict(X_test), 0, None)
    mae_hg = mean_absolute_error(y_test, preds)
    mae_tpha = mae_hg / 10000.0
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)

    return {
        "MAE_hg_ha": round(float(mae_hg), 2),
        "MAE_tpha": round(float(mae_tpha), 4),
        "RMSE": round(rmse, 2),
        "R2": round(float(r2), 4),
    }


def run_wandb_experiments(
    project_name: str = "crop-yield-mlops",
    raw_data_path: str = "data/raw/crop_yield_raw.csv",
    mode: str = "offline",
) -> Dict[str, Any]:
    """
    Runs candidate ML model experiments and logs metrics to Weights & Biases if available.
    """
    X, y = load_raw_crop_data(filepath=raw_data_path)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    experiments = [
        {
            "name": "Ridge_Baseline",
            "model": Ridge(alpha=1.0),
            "params": {"model_type": "Ridge", "alpha": 1.0},
        },
        {
            "name": "Random_Forest_Tuned",
            "model": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "params": {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 15},
        },
        {
            "name": "Gradient_Boosting",
            "model": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
            "params": {
                "model_type": "GradientBoosting",
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 5,
            },
        },
    ]

    runs_info = []

    for exp in experiments:
        run_name = exp["name"]
        raw_estimator = exp["model"]
        params = exp["params"]

        preprocessor = build_feature_preprocessor(random_state=42)
        inner_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", raw_estimator)])
        pipeline_model = TransformedTargetRegressor(
            regressor=inner_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        pipeline_model.fit(X_train, y_train)
        metrics = evaluate_wandb_model(pipeline_model, X_test, y_test)

        if HAS_WANDB:
            try:
                wandb.init(project=project_name, name=run_name, config=params, reinit=True, mode="disabled")
                wandb.log(metrics)
                wandb.finish()
            except Exception as e:
                logger.warning("wandb_logging_warning", error=str(e))

        runs_info.append(
            {
                "run_name": run_name,
                "model": pipeline_model,
                "metrics": metrics,
            }
        )

        logger.info("wandb_run_completed", run_name=run_name, mae_hg_ha=metrics["MAE_hg_ha"], r2=metrics["R2"])

    best_run = min(runs_info, key=lambda r: r["metrics"]["MAE_hg_ha"])
    save_artifacts(best_run["model"], pkl_path="models/model.pkl", onnx_path="models/model.onnx")

    return {
        "best_model_name": best_run["run_name"],
        "best_metrics": best_run["metrics"],
    }


if __name__ == "__main__":
    result = run_wandb_experiments()
    logger.info(
        "wandb_experiments_finished", best_model=result["best_model_name"], best_mae=result["best_metrics"]["MAE_hg_ha"]
    )
