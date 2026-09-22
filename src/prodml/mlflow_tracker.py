from typing import Any, Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from prodml.data import generate_synthetic_crop_data
from prodml.features import build_feature_preprocessor
from prodml.train import save_artifacts


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
) -> Dict[str, Any]:
    """
    Runs 3 distinct ML model experiments, logs metrics/params/artifacts to MLflow,
    and registers the best model to 'Staging'.
    """
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Generate synthetic dataset for reproducible tracking
    X, y = generate_synthetic_crop_data(n_samples=500, random_state=42)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Define 3 model candidates
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
        metrics = evaluate_model(pipeline_model, X_test, y_test)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("experiment_type", "crop_yield_pipeline")

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

            print(f"[OK] Completed Run: {run_name} | MAE (hg/ha): {metrics['MAE_hg_ha']} | R2: {metrics['R2']}")

    # Find best model (lowest MAE)
    best_run = min(runs_info, key=lambda r: r["metrics"]["MAE_hg_ha"])
    print(f"\n[BEST MODEL] {best_run['run_name']} with MAE: {best_run['metrics']['MAE_hg_ha']} hg/ha")

    # Register best model to MLflow Model Registry and assign stage 'Staging'
    model_uri = f"runs:/{best_run['run_id']}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    # Assign stage 'Staging' and alias 'Staging'
    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_details.version,
            stage="Staging",
            archive_existing_versions=True,
        )
    except Exception as e:
        print(f"Stage transition notice: {e}")

    try:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="Staging",
            version=model_details.version,
        )
    except Exception as e:
        print(f"Alias registration notice: {e}")

    # Export best model as primary local model artifact for API
    save_artifacts(best_run["model"], pkl_path="models/model.pkl", onnx_path="models/model.onnx")

    return {
        "best_run_id": best_run["run_id"],
        "best_model_name": best_run["run_name"],
        "best_metrics": best_run["metrics"],
        "registered_version": model_details.version,
    }


if __name__ == "__main__":
    print("Executing MLflow Experiment Runs & Staging Registration...")
    result = run_mlflow_experiments()
    print("\n--- MLflow Tracking Summary ---")
    print("Registered Model Name: CropYieldModel")
    print(f"Staging Model Version: {result['registered_version']}")
    print(f"Best Model Architecture: {result['best_model_name']}")
    print(f"Best MAE (hg/ha): {result['best_metrics']['MAE_hg_ha']}")
