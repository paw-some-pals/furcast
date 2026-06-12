"""
Random forest training for animal shelter outcome prediction
Supports classification (categorical y) and regression (numeric y).
(classification - for the outcome_type target, regression - for the time_in_shelter_days target)

To use for both the classification and regression tasks, simply call train_random_forest with the appropriate X and y:
model_clf = train_random_forest(X_train, y_outcome_type)       # classifier
model_reg = train_random_forest(X_train, y_time_in_shelter)    # regressor
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_random_forest_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Return an unfitted pipeline (for use with cross_val_score)."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ],
        remainder="passthrough",
    )
    is_regression = pd.api.types.is_numeric_dtype(y)
    estimator = (
        RandomForestRegressor(n_estimators=100, random_state=42)
        if is_regression
        else RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Train a random forest on features X and target y.

    Categorical columns in X are one-hot encoded automatically.
    Uses RandomForestRegressor when y is numeric, RandomForestClassifier otherwise.

    Returns the trained model.
    """
    model = build_random_forest_pipeline(X, y)
    model.fit(X, y)
    return model
