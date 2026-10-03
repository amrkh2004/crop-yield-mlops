"""
MLflow Experiment Tracking and Model Registry Management for Crop Yield Service.
Runs 8 candidate model & hyperparameter sweep experiments on real Kaggle crop yield data,
logs metrics/params/tags (including git_commit and sweep tags), registers top performing model
to MLflow Registry, and promotes to 'Production' stage.
"""

import subprocess
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from prodml.data import load_raw_crop_data
from prodml.features import build_feature_preprocessor
from prodml.logging import get_logger
from prodml.train import save_artifacts

logger = get_logger("prodml.mlflow_tracker")


def get_git_commit() -> str:
    """
    Helper function to get current HEAD git commit hash.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "unknown"


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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


def run_mlflow_experiments(
    experiment_name: str = "Crop_Yield_Prediction",
    registered_model_name: str = "CropYieldModel",
    raw_data_path: str = "data/raw/crop_yield_raw.csv",
    pkl_path: str = "models/model.pkl",
    onnx_path: str = "models/model.onnx",
) -> Dict[str, Any]:
    """
    Runs distinct ML candidate model & hyperparameter sweep experiments on real data,
    logs metrics, parameters, tags (including git_commit), and artifacts to MLflow,
    and promotes the top model to 'Production' stage in MLflow Model Registry.
    """
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    git_commit_hash = get_git_commit()

    # Load real Kaggle crop yield dataset with shuffled random split
    from sklearn.model_selection import train_test_split

    X, y = load_raw_crop_data(filepath=raw_data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Define candidate architectures & hyperparameter sweep variations
    experiments = [
        {
            "name": "Ridge_Baseline",
            "model": Ridge(alpha=1.0),
            "params": {"model_type": "Ridge", "alpha": 1.0},
        },
        {
            "name": "Linear_Regression",
            "model": LinearRegression(),
            "params": {"model_type": "LinearRegression"},
        },
        {
            "name": "Decision_Tree_Depth10",
            "model": DecisionTreeRegressor(max_depth=10, random_state=42),
            "params": {"model_type": "DecisionTree", "max_depth": 10},
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
        {
            "name": "Extra_Trees",
            "model": ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "params": {"model_type": "ExtraTrees", "n_estimators": 100, "max_depth": 15},
        },
        {
            "name": "Random_Forest_Sweep_Depth20",
            "model": RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1),
            "params": {"model_type": "RandomForest", "n_estimators": 150, "max_depth": 20, "sweep": True},
        },
        {
            "name": "Extra_Trees_Sweep_Depth20",
            "model": ExtraTreesRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1),
            "params": {"model_type": "ExtraTrees", "n_estimators": 150, "max_depth": 20, "sweep": True},
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
        metrics = evaluate_model(pipeline_model, X_test, y_test)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("experiment_type", "crop_yield_pipeline")
            mlflow.set_tag("dataset", "kaggle_crop_yield_real")
            mlflow.set_tag("author", "mlops_team")
            mlflow.set_tag("git_commit", git_commit_hash)

            mlflow.sklearn.log_model(
                sk_model=pipeline_model,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            runs_info.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run_name,
                    "model": pipeline_model,
                    "metrics": metrics,
                }
            )

            logger.info(
                "mlflow_run_completed",
                run_name=run_name,
                run_id=run.info.run_id,
                mae_hg_ha=metrics["MAE_hg_ha"],
                r2=metrics["R2"],
                git_commit=git_commit_hash,
            )

    # Best model selection (highest R2 / lowest MAE)
    best_run = max(runs_info, key=lambda r: r["metrics"]["R2"])
    logger.info(
        "best_model_selected",
        run_name=best_run["run_name"],
        mae_hg_ha=best_run["metrics"]["MAE_hg_ha"],
        r2=best_run["metrics"]["R2"],
    )

    # Register best model in MLflow Registry and transition stage to Production
    model_uri = f"runs:/{best_run['run_id']}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    # Promote model to 'Production' stage & assign 'Production' alias
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_details.version,
            stage="Production",
            archive_existing_versions=True,
        )
    except Exception as e:
        logger.warning("stage_transition_notice", error=str(e))

    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Production",
            version=model_details.version,
        )
    except Exception as e:
        logger.warning("alias_registration_notice", error=str(e))

    # Export best model as primary local model artifact
    save_artifacts(best_run["model"], pkl_path=pkl_path, onnx_path=onnx_path)

    return {
        "best_run_id": best_run["run_id"],
        "best_model_name": best_run["run_name"],
        "best_metrics": best_run["metrics"],
        "registered_version": model_details.version,
        "total_runs": len(experiments),
    }


if __name__ == "__main__":
    result = run_mlflow_experiments()
    logger.info(
        "mlflow_tracking_summary",
        registered_model="CropYieldModel",
        production_version=result["registered_version"],
        best_architecture=result["best_model_name"],
        best_mae=result["best_metrics"]["MAE_hg_ha"],
        total_runs=result["total_runs"],
    )
