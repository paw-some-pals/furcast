"""
Run training pipeline for the AAC (Austin Animal Center) dataset.

Assumes aac_cleaning.py has already been run and datasets/final_df_aac_dogs.csv exists.

Trains a Random Forest for:
  - outcome_type (classification)
  - time_in_shelter (regression)
  - stay_category (classification)

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

# Paths to the cleaned input data and the folder where trained models will be saved
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_df_aac_dogs.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_COLS = [ # MI with time_in_shelter
    "age_intake",           # 0.159
    "sex",                  # 0.014
    "spay_neuter",          # 0.069
    "intake_month",         # 0.079
    "intake_day",
    "intake_year",          # 0.049
    "animal_species",
    "animal_size",          # 0.058
    "colour",               # 0.016
    "breed",                # 0.154
    "intake_condition",     # 0.022
    "intake_type",          # 0.085
    "is_mixed",             # 0.003
    "breed_1",
    "breed_2",
    "good_with_children",   # 0.035
    "good_with_other_dogs", # 0.048
    "shedding",             # 0.044
    "grooming",
    "drooling",
    "coat_length",          # 0.020
    "good_with_strangers",  # 0.049
    "playfulness",          # 0.031
    "protectiveness",
    "trainability",         # 0.029
    "energy",               # 0.036
    "barking",              # 0.045
    "season",               # 0.040
    "population",           # 0.048
    "unemploy_rate",        # 0.085
]

TARGET_CLF  = "outcome_type"
TARGET_REG  = "time_in_shelter"
TARGET_CLF2 = "stay_category"

# def get_seasons(month):  # duplicate — season already computed in final_df_aac_dogs.csv
#     if month in [3,4,5]:
#         return "Spring"
#     elif month in [6,7,8]:
#         return "Summer"
#     elif month in [9,10,11]:
#         return "Fall"
#     elif month in [1,2,12]:
#         return "Winter"


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    print("=== Loading AAC cleaned dog data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    # df["season"] = df["intake_month"].apply(get_seasons)  # already in dataset
    print(f"{len(df)} rows loaded\n")

    X = df[FEATURE_COLS]

    # --- Model 1: outcome_type (classification) ---
    print("=== Training: outcome_type (classification) dog ===")
    y_clf = df[TARGET_CLF]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf), X, y_clf, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "aac_outcome_type_dog.pkl")

    # --- Model 2: time_in_shelter (regression) ---
    print("\n=== Training: time_in_shelter dog (regression) ===")

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
    save_model(model_reg, "aac_time_in_shelter_dog.pkl")

    # --- Model 3: stay_category (classification) ---
    print("\n=== Training: stay_category (classification) dog ===")
    y_clf2 = df[TARGET_CLF2]

    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf2), X, y_clf2, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_clf2, test_size=0.2, random_state=42)
    model_clf2 = train_random_forest(X_train, y_train)
    evaluate(model_clf2, X_test, y_test)
    save_model(model_clf2, "aac_stay_category_dog.pkl")


if __name__ == "__main__":
    main()
