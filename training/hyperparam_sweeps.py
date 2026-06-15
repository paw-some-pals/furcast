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
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
}



def build_tuned_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Build a pipeline with an ordinal encoder for categorical features and a random forest model.
    Categorical columns in X are one-hot encoded automatically.
    Uses RandomForestRegressor when y is numeric, RandomForestClassifier otherwise.
    Returns the un-fitted pipeline (GridSearchCV will fit it with different hyperparameters).
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
        RandomForestRegressor(random_state=42)
        if is_regression
        else RandomForestClassifier(random_state=42, class_weight="balanced")
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_random_forest_tuned(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Run a grid search over param_grid and return the best fitted pipeline.
    Uses shuffled folds so CV scores are comparable to a random train/test split.
    """
    # Determine if it's a regression or classification problem based on the target dtype
    is_regression = pd.api.types.is_numeric_dtype(y)
    # Use KFold for regression, StratifiedKFold for classification to maintain class balance in folds
    # put in manually to ensure shuffle=True and random_state=42 for reproducibility and comparability to random train/test split scores
    cv = (
        KFold(n_splits=5, shuffle=True, random_state=42)
        if is_regression
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    # Use appropriate scoring metric for regression vs classification
    scoring = "neg_mean_absolute_error" if is_regression else "accuracy"

    grid_search = GridSearchCV(
        estimator=build_tuned_pipeline(X, y),
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1, # Uses all available CPU cores for speed
        verbose=1, # Print progress of grid search to console
    )
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score:   {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_

