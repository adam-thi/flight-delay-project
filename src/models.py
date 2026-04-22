import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    n_estimators: int = 100,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="most_frequent"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_delay": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_delay": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_delay": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
