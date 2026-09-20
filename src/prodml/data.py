import numpy as np
import pandas as pd
from typing import Tuple, List

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


def generate_synthetic_crop_data(
    n_samples: int = 300, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a synthetic crop yield dataset matching notebook feature definitions.
    Features: Area, Item, Area_Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
    Target: hg/ha_yield.
    """
    areas = ["Albania", "Egypt", "India", "United States of America", "Brazil"]
    items = ["Maize", "Wheat", "Potatoes", "Rice, paddy", "Sorghum"]

    rng = np.random.RandomState(random_state)

    data = {
        "Area": rng.choice(areas, size=n_samples),
        "Item": rng.choice(items, size=n_samples),
        "Year": rng.randint(1990, 2024, size=n_samples),
        "average_rain_fall_mm_per_year": rng.uniform(300.0, 1800.0, size=n_samples),
        "pesticides_tonnes": rng.uniform(10.0, 500.0, size=n_samples),
        "avg_temp": rng.uniform(12.0, 35.0, size=n_samples),
    }

    df = pd.DataFrame(data)
    df["Area_Item"] = df["Area"] + "_" + df["Item"]

    yield_hg = (
        (df["average_rain_fall_mm_per_year"] * 15.0)
        + (df["pesticides_tonnes"] * 40.0)
        + (df["avg_temp"] * 300.0)
        + rng.normal(0, 2000, size=n_samples)
    )
    yield_hg = np.maximum(yield_hg, 1000.0)

    return df[FEATURE_NAMES], pd.Series(yield_hg, name=TARGET_NAME)
