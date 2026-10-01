"""
Dataset Downloader and Verifier for Crop Yield MLOps Service.
DVC is the primary versioning tool for data/raw/crop_yield_raw.csv.
"""

import os
import sys

import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "crop_yield_raw.csv")

REQUIRED_COLUMNS = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield",
]


def download_raw_dataset(output_path: str = RAW_DATA_PATH, force_download: bool = False) -> str:
    """
    Checks for raw dataset locally or instructs user/evaluator on DVC pull and Kaggle download.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force_download:
        df = pd.read_csv(output_path)
        print(f"[OK] Raw dataset already exists at '{output_path}' ({len(df):,} rows).")
        return output_path

    print(
        "[INFO] The raw dataset is versioned with DVC.\n"
        "To fetch the dataset, run:\n"
        "    dvc pull   (needs read access to the DVC remote, see README)\n"
        "Alternatively, place Kaggle 'Crop Yield Prediction' dataset (yield_df.csv) at 'data/raw/crop_yield_raw.csv'."
    )
    return output_path


def verify_dataset_integrity(filepath: str = RAW_DATA_PATH) -> bool:
    """
    Verifies column schema, row count, and non-empty status of the raw dataset.
    """
    if not os.path.exists(filepath):
        print(f"[WARN] Raw dataset file '{filepath}' does not exist locally.")
        print("Run: dvc pull   (see README for read-only DVC remote credentials)")
        return False

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[FAIL] Dataset is missing required columns: {missing_cols}")
        return False

    if len(df) < 1000:
        print(f"[WARN] Dataset row count is low ({len(df)} rows). Real dataset contains ~28,242 rows.")
        return False

    print(f"[VERIFIED] Dataset at '{filepath}' passed schema validation ({len(df):,} rows, {len(df.columns)} columns).")
    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = download_raw_dataset(force_download=force)
    verify_dataset_integrity(path)
