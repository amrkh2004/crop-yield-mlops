import glob
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from prodml.data import FEATURE_NAMES
from prodml.logging import get_logger

logger = get_logger("prodml.batch_score")


def get_production_model(model_name: str = "CropYieldModel"):
    """
    Attempts to load the Production model from MLflow Registry.
    Falls back to local model file or baseline model pipeline if MLflow is unreachable.
    """
    try:
        import mlflow.pyfunc

        model_uri = f"models:/{model_name}/Production"
        logger.info("fetching_production_model", uri=model_uri)
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.warning("mlflow_load_fallback", error=str(e))
        model_path = "models/model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            from prodml.train import train_model_pipeline

            return train_model_pipeline()


def run_batch_scoring(
    input_dir: str = "data/scoring/input",
    output_dir: str = "data/scoring/output",
    model_name: str = "CropYieldModel",
) -> str:
    """
    Executes batch prediction on all Parquet files in input_dir,
    appends run_date, and writes output Parquet files to output_dir.
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))

    if not input_files:
        sample_file = os.path.join(input_dir, "sample_crops.parquet")
        areas = ["Egypt", "India", "Brazil", "United States of America", "Albania"]
        items = ["Potatoes", "Wheat", "Maize", "Rice, paddy", "Sorghum"]
        rng = np.random.RandomState(42)

        chosen_areas = rng.choice(areas, size=100)
        chosen_items = rng.choice(items, size=100)

        sample_data = pd.DataFrame(
            {
                "crop_id": [f"crop_{i:04d}" for i in range(1, 101)],
                "Area": chosen_areas,
                "Item": chosen_items,
                "Area_Item": [f"{a}_{i}" for a, i in zip(chosen_areas, chosen_items)],
                "Year": rng.randint(1990, 2024, size=100),
                "average_rain_fall_mm_per_year": rng.uniform(300.0, 1800.0, size=100).round(2),
                "pesticides_tonnes": rng.uniform(10.0, 500.0, size=100).round(2),
                "avg_temp": rng.uniform(12.0, 35.0, size=100).round(2),
            }
        )
        sample_data.to_parquet(sample_file, index=False)
        input_files = [sample_file]
        logger.info("generated_sample_input", path=sample_file)

    model = get_production_model(model_name)
    run_date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    processed_count = 0
    latest_output_path = ""

    for file_path in input_files:
        df = pd.read_parquet(file_path)
        logger.info("processing_batch_file", file=file_path, records=len(df))

        if "Area_Item" not in df.columns:
            df["Area_Item"] = df["Area"] + "_" + df["Item"]

        scoring_df = df[FEATURE_NAMES]

        if hasattr(model, "predict"):
            preds = model.predict(scoring_df)
        else:
            preds = model(scoring_df)

        preds_clean = np.clip(preds, 0, None)
        df["predicted_yield_hg_ha"] = np.round(preds_clean, 2)
        df["predicted_yield_tons_ha"] = np.round(preds_clean / 10000.0, 4)
        df["run_date"] = run_date_str

        base_name = os.path.basename(file_path).replace(".parquet", "")
        timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_scored_{timestamp_suffix}.parquet"
        output_file_path = os.path.join(output_dir, output_filename)

        df.to_parquet(output_file_path, index=False)
        logger.info("batch_scored_file_saved", path=output_file_path)
        latest_output_path = output_file_path
        processed_count += len(df)

    logger.info("batch_scoring_completed", processed_count=processed_count)
    return latest_output_path


if __name__ == "__main__":
    run_batch_scoring()
