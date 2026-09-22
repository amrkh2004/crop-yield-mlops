import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import wandb
from prodml.data import generate_synthetic_crop_data
from prodml.features import build_feature_preprocessor


def evaluate_wandb_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluates predictions for W&B logging.
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
    mode: str = None,
) -> List[Dict[str, Any]]:
    """
    Runs candidate model experiments and logs metrics/configs to Weights & Biases (W&B).
    Supports offline mode (WANDB_MODE=offline) for seamless local execution.
    """
    wandb_mode = mode or os.getenv("WANDB_MODE", "offline")
    os.environ["WANDB_MODE"] = wandb_mode

    X, y = generate_synthetic_crop_data(n_samples=500, random_state=42)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    experiments = [
        {
            "name": "Ridge_Baseline",
            "model": Ridge(alpha=1.0),
            "config": {"architecture": "Ridge", "alpha": 1.0, "framework": "scikit-learn"},
        },
        {
            "name": "Random_Forest_Tuned",
            "model": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "config": {
                "architecture": "RandomForest",
                "n_estimators": 100,
                "max_depth": 15,
                "framework": "scikit-learn",
            },
        },
        {
            "name": "Gradient_Boosting",
            "model": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
            "config": {
                "architecture": "GradientBoosting",
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 5,
                "framework": "scikit-learn",
            },
        },
    ]

    results = []

    for exp in experiments:
        run_name = exp["name"]
        raw_estimator = exp["model"]
        config = exp["config"]

        preprocessor = build_feature_preprocessor(random_state=42)
        inner_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", raw_estimator)])
        pipeline_model = TransformedTargetRegressor(
            regressor=inner_pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )

        pipeline_model.fit(X_train, y_train)
        metrics = evaluate_wandb_model(pipeline_model, X_test, y_test)

        try:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True,
                mode=wandb_mode,
            )
            wandb.log(metrics)
            wandb.summary["best_MAE_hg_ha"] = metrics["MAE_hg_ha"]
            wandb.summary["best_R2"] = metrics["R2"]
            wandb.finish()
        except Exception as e:
            print(f"[W&B NOTICE] Running in fallback mode: {e}")

        results.append(
            {
                "run_name": run_name,
                "config": config,
                "metrics": metrics,
            }
        )

        print(f"[OK W&B] Logged Run: {run_name} | MAE (hg/ha): {metrics['MAE_hg_ha']} | R2: {metrics['R2']}")

    best_run = min(results, key=lambda r: r["metrics"]["MAE_hg_ha"])
    print(f"\n[W&B BEST MODEL] {best_run['run_name']} with MAE: {best_run['metrics']['MAE_hg_ha']} hg/ha")

    return results


if __name__ == "__main__":
    print("Executing Weights & Biases (W&B) Experiment Runs...")
    res = run_wandb_experiments()
    print("\n--- Weights & Biases (W&B) Tracking Summary ---")
    for r in res:
        print(f"- Run: {r['run_name']:<20} | MAE: {r['metrics']['MAE_hg_ha']:<10} | R2: {r['metrics']['R2']}")
