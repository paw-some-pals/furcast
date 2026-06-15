"""
Step 1 of 2: Hyperparameter sweep for AAC dog models.

Runs a cross-validated grid search on the train split to find the best
hyperparameters for each target, then evaluates on the held-out test set.

Saves best params to models/aac_best_params_dog.json.
Run run_final_training_aac_dog.py after this to train on all data and save models.
"""

import sys
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # training/ for evaluate.py

from hyperparam_sweeps import train_random_forest_tuned
from evaluate import evaluate

CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "final_df_aac_dogs.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PARAMS_PATH = os.path.join(MODELS_DIR, "aac_best_params_dog.json")

FEATURE_COLS = [
    "age_intake",
    "sex",
    "spay_neuter",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    "animal_size",
    "colour",
    "breed",
    "intake_condition",
    "intake_type",
    "is_mixed",
    "breed_1",
    "breed_2",
    "good_with_children",
    "good_with_other_dogs",
    "shedding",
    "grooming",
    "drooling",
    "coat_length",
    "good_with_strangers",
    "playfulness",
    "protectiveness",
    "trainability",
    "energy",
    "barking",
    "season",
    "population",
    "unemploy_rate",
]

TARGET_CLF  = "outcome_type"
TARGET_REG  = "time_in_shelter"
TARGET_CLF2 = "stay_category"

SWEPT_PARAMS = ["n_estimators", "max_depth", "min_samples_split"]


def extract_best_params(model):
    """Pull the swept hyperparameter values out of a fitted pipeline."""
    rf = model.named_steps["model"]
    return {f"model__{p}": getattr(rf, p) for p in SWEPT_PARAMS}


def main():
    print("=== Loading AAC cleaned dog data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"{len(df)} rows loaded\n")

    X = df[FEATURE_COLS]
    best_params = {}

    # --- outcome_type (classification) ---
    print("=== Sweep: outcome_type (classification) ===")
    y = df[TARGET_CLF]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_random_forest_tuned(X_train, y_train)
    print("\n--- Held-out test set evaluation ---")
    evaluate(model, X_test, y_test)
    best_params[TARGET_CLF] = extract_best_params(model)
    print(f"Best params: {best_params[TARGET_CLF]}")

    # --- time_in_shelter (regression) ---
    print("\n=== Sweep: time_in_shelter (regression) ===")
    y = np.log1p(df[TARGET_REG])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_random_forest_tuned(X_train, y_train)
    print("\n--- Held-out test set evaluation ---")
    y_pred_days = np.expm1(model.predict(X_test))
    y_test_days = np.expm1(y_test)
    print(f"MAE (days):  {mean_absolute_error(y_test_days, y_pred_days):.2f}")
    print(f"RMSE (days): {root_mean_squared_error(y_test_days, y_pred_days):.2f}")
    best_params[TARGET_REG] = extract_best_params(model)
    print(f"Best params: {best_params[TARGET_REG]}")

    # --- stay_category (classification) ---
    print("\n=== Sweep: stay_category (classification) ===")
    y = df[TARGET_CLF2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_random_forest_tuned(X_train, y_train)
    print("\n--- Held-out test set evaluation ---")
    evaluate(model, X_test, y_test)
    best_params[TARGET_CLF2] = extract_best_params(model)
    print(f"Best params: {best_params[TARGET_CLF2]}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {PARAMS_PATH}")
    print("Now run run_final_training_aac_dog.py to train on all data and save models.")


if __name__ == "__main__":
    main()
