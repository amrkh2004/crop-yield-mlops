"""
Dataset Downloader and Verifier for Crop Yield MLOps Service.
Downloads real Kaggle Crop Yield raw dataset (25,932 rows) to data/raw/crop_yield_raw.csv.
"""

import os
import sys
import urllib.request

import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "crop_yield_raw.csv")
DATASET_URL = "https://raw.githubusercontent.com/amrkh2004/crop-yield-mlops/main/data/raw/crop_yield_raw.csv"

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
    Downloads raw crop yield dataset from repository mirror if not present locally.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not force_download:
        df = pd.read_csv(output_path)
        print(f"[OK] Raw dataset already exists at '{output_path}' ({len(df):,} rows).")
        return output_path

    print(f"[FETCH] Downloading raw crop yield dataset from '{DATASET_URL}'...")
    try:
        urllib.request.urlretrieve(DATASET_URL, output_path)
        df = pd.read_csv(output_path)
        print(f"[SUCCESS] Downloaded {len(df):,} rows to '{output_path}'.")
        return output_path
    except Exception as e:
        print(f"[ERROR] Failed to download dataset online: {e}")
        if os.path.exists(output_path):
            print(f"[INFO] Using existing file at '{output_path}'.")
            return output_path
        raise RuntimeError(f"Could not retrieve raw dataset: {e}")


def verify_dataset_integrity(filepath: str = RAW_DATA_PATH) -> bool:
    """
    Verifies column schema, row count, and non-empty status of the raw dataset.
    """
    if not os.path.exists(filepath):
        print(f"[FAIL] Raw dataset file '{filepath}' does not exist.")
        return False

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[FAIL] Dataset is missing required columns: {missing_cols}")
        return False

    if len(df) < 1000:
        print(f"[WARN] Dataset row count is low ({len(df)} rows). Real dataset contains ~25,932 rows.")
        return False

    print(f"[VERIFIED] Dataset at '{filepath}' passed schema validation ({len(df):,} rows, {len(df.columns)} columns).")
    return True


if __name__ == "__main__":
    force = "--force" in sys.argv
    path = download_raw_dataset(force_download=force)
    verify_dataset_integrity(path)
