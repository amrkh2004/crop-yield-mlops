import os
from typing import Tuple

import pandas as pd

from prodml.data import TARGET_NAME, load_raw_crop_data
from prodml.logging import get_logger

logger = get_logger("prodml.pipeline.prepare")


def run_prepare(
    raw_csv_path: str = "data/raw/crop_yield_raw.csv",
    output_dir: str = "data/prepared",
) -> Tuple[str, str]:
    """
    DVC Pipeline Stage 1: prepare
    Reads raw dataset, removes duplicates, adds Area_Item feature,
    and splits into train.csv and test.csv.
    """
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(raw_csv_path):
        df = pd.read_csv(raw_csv_path)
    else:
        X_raw, y_raw = load_raw_crop_data(filepath=raw_csv_path)
        df = X_raw.copy()
        df[TARGET_NAME] = y_raw
        df.to_csv(raw_csv_path, index=False)

    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    if "Area_Item" not in df.columns:
        df["Area_Item"] = df["Area"] + "_" + df["Item"]

    # Random train/test split with shuffle=True to prevent alphabetical country segregation
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        "dvc_prepare_completed",
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_path=train_path,
        test_path=test_path,
    )
    return train_path, test_path


if __name__ == "__main__":
    run_prepare()
