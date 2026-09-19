from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from prodml.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_feature_preprocessor() -> ColumnTransformer:
    """
    Builds and returns the scikit-learn ColumnTransformer for feature preprocessing:
    - StandardScaler for numeric features
    - OneHotEncoder (handle_unknown='ignore') for categorical features
    """
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
