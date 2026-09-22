import os
from typing import Tuple

import pandas as pd

from prodml.data import TARGET_NAME, generate_synthetic_crop_data


def run_prepare(
    raw_csv_path: str = "data/raw/crop_yield_raw.csv",
    output_dir: str = "data/prepared",
) -> Tuple[str, str]:
    """
    DVC Pipeline Stage 1: prepare
    Reads/generates raw dataset, removes duplicates, adds Area_Item feature,
    and splits into train.csv and test.csv.
    """
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(raw_csv_path):
        df = pd.read_csv(raw_csv_path)
    else:
        X_raw, y_raw = generate_synthetic_crop_data(n_samples=600, random_state=42)
        df = X_raw.copy()
        df[TARGET_NAME] = y_raw
        df.to_csv(raw_csv_path, index=False)

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    df["Area_Item"] = df["Area"] + "_" + df["Item"]

    # Time-based or stratified split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[DVC PREPARE] Saved train set ({len(train_df)} rows) to: {train_path}")
    print(f"[DVC PREPARE] Saved test set ({len(test_df)} rows) to: {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    run_prepare()
