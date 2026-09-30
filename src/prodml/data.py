"""
Data management module for Crop Yield Prediction dataset.
Provides real dataset loading from Kaggle raw CSV and synthetic fallback dataset generator.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from prodml.logging import get_logger

logger = get_logger("prodml.data")

FEATURE_NAMES: List[str] = [
    "Area",
    "Item",
    "Area_Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]

CATEGORICAL_FEATURES: List[str] = ["Area", "Item", "Area_Item"]
NUMERIC_FEATURES: List[str] = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
TARGET_NAME: str = "hg/ha_yield"


def load_raw_crop_data(
    filepath: str = "data/raw/crop_yield_raw.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads real Kaggle crop yield dataset from raw CSV.
    Ensures Area_Item interaction feature exists and returns features (X) and target (y).
    """
    if not os.path.exists(filepath):
        logger.warning("raw_file_not_found", path=filepath, message="Falling back to synthetic data generation")
        return generate_synthetic_crop_data(n_samples=600, random_state=42)

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    if "Area_Item" not in df.columns and "Area" in df.columns and "Item" in df.columns:
        df["Area_Item"] = df["Area"] + "_" + df["Item"]

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    logger.info("real_data_loaded", rows=len(df), features=len(FEATURE_NAMES), target=TARGET_NAME)
    return X, y


def generate_synthetic_crop_data(n_samples: int = 600, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates realistic crop yield dataset incorporating area and crop item multipliers.
    """
    areas = ["Albania", "Egypt", "India", "United States of America", "Brazil"]
    items = ["Maize", "Wheat", "Potatoes", "Rice, paddy", "Sorghum"]

    area_effects = {
        "Albania": 15000.0,
        "Egypt": 32000.0,
        "India": 25000.0,
        "United States of America": 42000.0,
        "Brazil": 28000.0,
    }
    item_effects = {
        "Maize": 12000.0,
        "Wheat": 8000.0,
        "Potatoes": 35000.0,
        "Rice, paddy": 22000.0,
        "Sorghum": 14000.0,
    }

    rng = np.random.RandomState(random_state)

    chosen_areas = rng.choice(areas, size=n_samples)
    chosen_items = rng.choice(items, size=n_samples)
    years = rng.randint(1990, 2024, size=n_samples)
    rain = rng.uniform(300.0, 1800.0, size=n_samples)
    pesticides = rng.uniform(10.0, 500.0, size=n_samples)
    temp = rng.uniform(12.0, 35.0, size=n_samples)

    df = pd.DataFrame(
        {
            "Area": chosen_areas,
            "Item": chosen_items,
            "Year": years,
            "average_rain_fall_mm_per_year": rain,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp,
        }
    )
    df["Area_Item"] = df["Area"] + "_" + df["Item"]

    base_yield = np.array([area_effects[a] + item_effects[i] for a, i in zip(chosen_areas, chosen_items)])
    yield_hg = base_yield + (rain * 5.0) + (pesticides * 20.0) + (temp * 150.0) + rng.normal(0, 3000, size=n_samples)
    yield_hg = np.maximum(yield_hg, 1000.0)

    return df[FEATURE_NAMES], pd.Series(yield_hg, name=TARGET_NAME)
