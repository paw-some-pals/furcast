"""
XGBoost training for animal shelter outcome prediction
Supports classification (categorical y) and regression (numeric y).
(classification - for the outcome_type target, regression - for the time_in_shelter_days target)

To use for both the classification and regression tasks, simply call train_xgboost with the appropriate X and y:
model_clf = train_xgboost(X_train, y_outcome_type)       # classifier
model_reg = train_xgboost(X_train, y_time_in_shelter)    # regressor
"""

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class CategoricalCaster(BaseEstimator, TransformerMixin):
    """Casts object/category columns to pandas Categorical dtype for XGBoost native categorical support."""

    def fit(self, X, _y=None):
        self.categorical_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.categorical_cols_] = X[self.categorical_cols_].astype("category")
        return X


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Train an XGBoost model on features X and target y.

    Categorical columns are cast to pandas Categorical dtype so XGBoost can
    find optimal splits natively (enable_categorical=True).
    Uses XGBRegressor when y is numeric, XGBClassifier otherwise.

    Returns the trained pipeline.
    """
    is_regression = pd.api.types.is_numeric_dtype(y)
    estimator = (
        XGBRegressor(enable_categorical=True, tree_method="hist", n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0)
        if is_regression
        else XGBClassifier(enable_categorical=True, tree_method="hist", n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0, eval_metric="mlogloss")
    )

    model = Pipeline([("caster", CategoricalCaster()), ("model", estimator)])
    model.fit(X, y)

    return model
