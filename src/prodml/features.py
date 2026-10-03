import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)


def build_feature_preprocessor(random_state: int = 42) -> ColumnTransformer:
    """
    Builds and returns the scikit-learn ColumnTransformer for feature preprocessing:
    - StandardScaler for numeric features (Year, rainfall, temp)
    - log1p + StandardScaler for pesticides
    - OneHotEncoder (handle_unknown='ignore') for Area and Item
    - TargetEncoder for high-cardinality Area_Item interaction feature
    """
    onehot_features = ["Area", "Item"]
    te_features = ["Area_Item"]
    num_features = ["Year", "average_rain_fall_mm_per_year", "avg_temp"]
    log_features = ["pesticides_tonnes"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "pest",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                log_features,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                onehot_features,
            ),
            (
                "te",
                TargetEncoder(target_type="continuous", cv=5, random_state=random_state),
                te_features,
            ),
        ]
    )


CAT_COLS = ["Area", "Item", "Area_Item"]
DL_NUM = ["Year", "average_rain_fall_mm_per_year", "avg_temp"]


def _num_matrix(df) -> np.ndarray:
    return np.column_stack(
        [
            df[DL_NUM].values.astype("float64"),
            np.log1p(df["pesticides_tonnes"].values.astype("float64")),
        ]
    )


def fit_dl_preprocessing(df, target_col: str = "hg/ha_yield") -> dict:
    import pandas as pd

    df_copy = pd.DataFrame(df).copy()
    if "Area_Item" not in df_copy.columns:
        df_copy["Area_Item"] = df_copy["Area"].astype(str) + "_" + df_copy["Item"].astype(str)

    vocabs = {}
    for c in CAT_COLS:
        unique_vals = sorted(df_copy[c].astype(str).unique())
        vocabs[c] = {v: i + 1 for i, v in enumerate(unique_vals)}

    scaler = StandardScaler().fit(_num_matrix(df_copy))

    if target_col in df_copy.columns:
        y_log = np.log1p(df_copy[target_col].values.astype("float64"))
        y_mean = float(y_log.mean())
        y_std = float(y_log.std()) if float(y_log.std()) > 1e-6 else 1.0
    else:
        y_mean, y_std = 0.0, 1.0

    return {
        "vocabs": vocabs,
        "scaler": scaler,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def transform_dl_inputs(df, prep: dict) -> dict:
    import pandas as pd

    df_copy = pd.DataFrame(df).copy()
    if "Area_Item" not in df_copy.columns:
        df_copy["Area_Item"] = df_copy["Area"].astype(str) + "_" + df_copy["Item"].astype(str)

    X = {}
    for c in CAT_COLS:
        mapped = df_copy[c].astype(str).map(prep["vocabs"][c]).fillna(0).astype("int32").values
        X[f"{c}_in"] = mapped.reshape(-1, 1)

    X["num_in"] = prep["scaler"].transform(_num_matrix(df_copy)).astype("float32")
    return X


def transform_dl_target(y: np.ndarray, prep: dict) -> np.ndarray:
    y_log = np.log1p(np.asarray(y, dtype="float64"))
    return ((y_log - prep["y_mean"]) / prep["y_std"]).astype("float32")


def inverse_dl_target(z: np.ndarray, prep: dict) -> np.ndarray:
    z_arr = np.asarray(z, dtype="float64").reshape(-1)
    y_log = z_arr * prep["y_std"] + prep["y_mean"]
    return np.expm1(y_log)
