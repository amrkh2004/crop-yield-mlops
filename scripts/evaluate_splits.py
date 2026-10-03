"""
Compare train/test splitting strategies on the real Kaggle crop yield dataset.

Usage (from the repo root):
    python scripts/evaluate_splits.py

Uses the same preprocessing + Gradient Boosting pipeline as the DVC `train` stage, so the
"random shuffle" row reproduces reports/metrics.json.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from prodml.data import FEATURE_NAMES, TARGET_NAME
from prodml.features import build_feature_preprocessor

RAW_PATH = "data/raw/crop_yield_raw.csv"


def build_model() -> TransformedTargetRegressor:
    inner = Pipeline(
        steps=[
            ("prep", build_feature_preprocessor(random_state=42)),
            (
                "model",
                GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
            ),
        ]
    )
    return TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)


def evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = build_model().fit(train[FEATURE_NAMES], train[TARGET_NAME])
    preds = np.clip(model.predict(test[FEATURE_NAMES]), 0, None)
    mae = float(mean_absolute_error(test[TARGET_NAME], preds))
    r2 = float(r2_score(test[TARGET_NAME], preds))
    print(f"{name:38s} n_test={len(test):5d}  MAE={mae:9.0f} hg/ha ({mae / 10000:.2f} t/ha)  R2={r2:7.4f}")
    return {"split": name, "n_test": len(test), "MAE_hg_ha": round(mae, 2), "R2": round(r2, 4)}


def main() -> None:
    warnings.filterwarnings("ignore")
    df = pd.read_csv(RAW_PATH).drop_duplicates().reset_index(drop=True)
    df["Area_Item"] = df["Area"] + "_" + df["Item"]

    cut = int(len(df) * 0.8)
    evaluate("ordered 80/20 (no shuffle)", df.iloc[:cut], df.iloc[cut:])

    train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    evaluate("random shuffle (DVC pipeline)", train, test)

    evaluate("temporal (train<=2008, test>2008)", df[df["Year"] <= 2008], df[df["Year"] > 2008])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["Area"]))
    evaluate("unseen countries (group split)", df.iloc[train_idx], df.iloc[test_idx])


if __name__ == "__main__":
    main()
