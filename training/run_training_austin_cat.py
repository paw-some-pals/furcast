"""
Run training pipeline for the AAC (Austin Animal Center) dataset.

Assumes aac_cleaning.py has already been run and datasets/final_df_aac_cats.csv exists.

Trains a Random Forest for:
  - outcome_type (classification)
  - time_in_shelter (regression)

Saves trained models to models/.
"""

import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold, StratifiedKFold

# Allow importing from this same folder (training/)
sys.path.insert(0, os.path.dirname(__file__))

from random_forest_training import build_random_forest_pipeline, train_random_forest
from evaluate import evaluate

CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_df_aac_cats.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [ # MI with time_in_shelter (from cat heatmap)
    "age_intake",
    "sex",
    "spay_neuter",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    "animal_size",
    "colour",               # 0.048
    "breed",                # 0.042
    "intake_condition",     # 0.042
    "intake_type",          # 0.103
    "is_mixed",             # 0.007
    "breed_1",
    "breed_2",
    "min_life_expectancy",  # 0.017
    "max_life_expectancy",  # 0.010
    "min_weight",           # 0.010
    "max_weight",           # 0.010
    "family_friendly",      # 0.007
    "shedding",             # 0.006
    "general_health",       # 0.009
    "playfulness",          # 0.005
    "children_friendly",    # 0.006
    "grooming",             # 0.007
    "intelligence",         # 0.004
    "other_pets_friendly",  # 0.008
    "season",               # 0.139
    "black",
    "white",
    "population",
    "unemploy_rate",
]

TARGET_CLF = "outcome_type"
TARGET_REG = "time_in_shelter"


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    print("=== Loading AAC cleaned cat data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"{len(df)} rows loaded\n")

    X = df[FEATURE_COLS]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    # --- Model 1: outcome_type (classification) ---
    print("=== Training: outcome_type (classification) cat ===")
    y_clf = df[TARGET_CLF]

    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf), X, y_clf, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "aac_outcome_type_cat.pkl")

    # --- Model 2: time_in_shelter (regression) ---
    print("\n=== Training: time_in_shelter cat (regression) ===")

    # log1p applied because time_in_shelter is right-skewed; expm1 reverses it for interpretable error metrics
    y_reg = np.log1p(df[TARGET_REG])

    cv_preds = cross_val_predict(build_random_forest_pipeline(X, y_reg), X, y_reg, cv=kf)
    cv_mae = mean_absolute_error(np.expm1(y_reg), np.expm1(cv_preds))
    print(f"CV MAE: {cv_mae:.4f} days")

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    model_reg = train_random_forest(X_train, y_train)

    y_pred = np.expm1(model_reg.predict(X_test))
    y_test_orig = np.expm1(y_test)
    print(f"MAE:  {mean_absolute_error(y_test_orig, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test_orig, y_pred):.2f}")
    save_model(model_reg, "aac_time_in_shelter_cat.pkl")


if __name__ == "__main__":
    main()
