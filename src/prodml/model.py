import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


class CropYieldModel:
    """
    ML Pipeline wrapper for Crop Yield Prediction based on notebook implementation.
    Features: Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.
    Target: hg/ha_yield (hectograms per hectare).
    """

    FEATURE_NAMES = [
        "Area",
        "Item",
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]

    CATEGORICAL_FEATURES = ["Area", "Item"]
    NUMERIC_FEATURES = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]

    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = model_path
        self.pipeline: Pipeline = None

    def load_or_create(self) -> None:
        """
        Loads the trained model pipeline from disk if available,
        otherwise creates, trains, and saves a baseline model.
        """
        # Also check alternative path from notebook if model.pkl isn't found
        alt_path = "crop_yield_pipeline.joblib"
        target_file = self.model_path if os.path.exists(self.model_path) else (
            alt_path if os.path.exists(alt_path) else None
        )

        if target_file:
            self.pipeline = joblib.load(target_file)
        else:
            self.pipeline = self._create_and_train_baseline()
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)

    def _create_and_train_baseline(self) -> Pipeline:
        """
        Generates a baseline pipeline matching the notebook structure.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    self.NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    self.CATEGORICAL_FEATURES,
                ),
            ]
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=50, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        )

        # Baseline synthetic dataset matching notebook features
        areas = ["Albania", "Egypt", "India", "United States of America", "Brazil"]
        items = ["Maize", "Wheat", "Potatoes", "Rice, paddy", "Sorghum"]
        np.random.seed(42)
        n_samples = 250

        data = {
            "Area": np.random.choice(areas, size=n_samples),
            "Item": np.random.choice(items, size=n_samples),
            "Year": np.random.randint(1990, 2024, size=n_samples),
            "average_rain_fall_mm_per_year": np.random.uniform(300.0, 1800.0, size=n_samples),
            "pesticides_tonnes": np.random.uniform(10.0, 500.0, size=n_samples),
            "avg_temp": np.random.uniform(12.0, 35.0, size=n_samples),
        }

        df = pd.DataFrame(data)

        # Synthetic target (hg/ha_yield) matching real dataset range
        yield_hg = (
            (df["average_rain_fall_mm_per_year"] * 15.0)
            + (df["pesticides_tonnes"] * 40.0)
            + (df["avg_temp"] * 300.0)
            + np.random.normal(0, 2000, size=n_samples)
        )
        yield_hg = np.maximum(yield_hg, 1000.0)

        pipeline.fit(df[self.FEATURE_NAMES], yield_hg)
        return pipeline

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Runs model prediction for a set of crop/weather features.
        Returns predicted yield in hg/ha and converted metric tons/ha.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Call load_or_create() first.")

        df = pd.DataFrame([input_data])
        raw_pred = float(self.pipeline.predict(df[self.FEATURE_NAMES])[0])
        predicted_hg_ha = max(round(raw_pred, 2), 0.0)
        # 10,000 hg = 1 ton
        predicted_tons_ha = round(predicted_hg_ha / 10000.0, 2)

        return {
            "predicted_yield_hg_ha": predicted_hg_ha,
            "predicted_yield_tons_ha": predicted_tons_ha,
        }
