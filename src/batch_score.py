import os
import sys
import glob
from datetime import datetime
import pandas as pd
import numpy as np


def get_production_model(model_name: str = "RideDurationModel"):
    """
    Attempts to load the Production model from MLflow Registry.
    Falls back to a local model file or baseline Scikit-Learn pipeline if MLflow is unreachable.
    """
    try:
        import mlflow.pyfunc
        model_uri = f"models:/{model_name}/Production"
        print(f"[BatchScorer] Fetching production model from MLflow Registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"[BatchScorer] MLflow load failed ({e}). Using baseline RideDurationModel pipeline.")
        return BaselineRideDurationModel()


class BaselineRideDurationModel:
    """
    Fallback baseline model for ride-duration prediction.
    Formula: duration_minutes = 3.0 + (distance_km * 2.5) + (passengers * 0.5) + (hour_of_day * 0.2)
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        distance = df.get("distance_km", 5.0)
        passengers = df.get("passengers", 1)
        hour = df.get("hour_of_day", 12)
        return 3.0 + (distance * 2.5) + (passengers * 0.5) + (hour * 0.2)


def run_batch_scoring(
    input_dir: str = "data/scoring/input",
    output_dir: str = "data/scoring/output",
    model_name: str = "RideDurationModel"
) -> str:
    """
    Executes batch prediction on all Parquet files in input_dir,
    appends run_date, and writes output Parquet files to output_dir.
    """
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    # Generate sample input if no Parquet files exist
    if not input_files:
        sample_file = os.path.join(input_dir, "sample_rides.parquet")
        sample_data = pd.DataFrame({
            "ride_id": [f"ride_{i:04d}" for i in range(1, 101)],
            "distance_km": np.random.uniform(1.0, 30.0, size=100).round(2),
            "passengers": np.random.randint(1, 5, size=100),
            "hour_of_day": np.random.randint(0, 24, size=100)
        })
        sample_data.to_parquet(sample_file, index=False)
        input_files = [sample_file]
        print(f"[BatchScorer] Generated sample input Parquet file: {sample_file}")

    model = get_production_model(model_name)
    run_date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    processed_count = 0
    latest_output_path = ""

    for file_path in input_files:
        df = pd.read_parquet(file_path)
        print(f"[BatchScorer] Processing {len(df)} records from {file_path}")

        # Execute prediction
        if hasattr(model, "predict"):
            preds = model.predict(df)
        else:
            preds = model(df)

        df["predicted_duration_minutes"] = np.round(preds, 2)
        df["run_date"] = run_date_str

        # Define output filepath
        base_name = os.path.basename(file_path).replace(".parquet", "")
        timestamp_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_scored_{timestamp_suffix}.parquet"
        output_file_path = os.path.join(output_dir, output_filename)

        df.to_parquet(output_file_path, index=False)
        print(f"[BatchScorer] Scored file saved: {output_file_path}")
        latest_output_path = output_file_path
        processed_count += len(df)

    print(f"[BatchScorer] Successfully scored {processed_count} total records.")
    return latest_output_path


if __name__ == "__main__":
    run_batch_scoring()
