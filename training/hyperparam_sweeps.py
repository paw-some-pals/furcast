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
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Train a random forest on features X and target y.

    Categorical columns in X are one-hot encoded automatically.
    Uses RandomForestRegressor when y is numeric, RandomForestClassifier otherwise.

    Returns the trained model.
    """
    # Identify categorical columns (object or category dtype)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # One-hot encode categorical features, passthrough numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols),
        ],
        remainder="passthrough",
    )

    # Determine if it's a regression or classification problem based on the target dtype
    is_regression = pd.api.types.is_numeric_dtype(y)
    # Choose the appropriate random forest estimator
    estimator = (
        RandomForestRegressor(n_estimators=100, random_state=42)
        if is_regression
        else RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    )

    # Create a pipeline that first preprocesses the data, then fits the model
    model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    grid_search = GridSearchCV(
                    estimator=model, 
                    param_grid=param_grid, 
                    cv=5, 
                    scoring='accuracy', 
                    n_jobs=-1  # Uses all available CPU cores for speed
                )
    model.fit(X, y)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    return model
