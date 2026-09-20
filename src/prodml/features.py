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
