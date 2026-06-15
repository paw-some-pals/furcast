"""
Step 2 of 2: Train AAC cat models on all data using best hyperparameters from the sweep.

Run run_sweep_aac_cat.py first to generate models/aac_best_params_cat.json.
Saves tuned models to models/aac_*_cat_tuned.pkl.
"""

import sys
import os
import json
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from hyper_param_sweeps.hyperparam_sweeps import train_final_model

CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "ABDLT_output_cat_pop.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PARAMS_PATH = os.path.join(MODELS_DIR, "abdlt_best_params_cat.json")

FEATURE_COLS = [
    "age_intake",
    "sex",
    #"spay_neuter",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    #"animal_size",
    "colour",
    #"breed",
    #"intake_condition",
    "intake_type",
    "is_mixed",
    "breed_1",
    "breed_2",
    #"min_life_expectancy",
    #"max_life_expectancy",
    #"min_weight",
    #"max_weight",
    "family_friendly",
    "shedding",
    "general_health",
    "playfulness",
    "children_friendly",
    "grooming",
    "intelligence",
    "other_pets_friendly",
    "season",
    "black",
    "white",
    "population",
    "unemploy_rate",
]

TARGET_CLF  = "outcome_type"
TARGET_REG  = "time_in_shelter"
#TARGET_CLF2 = "stay_category"


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(
            f"Best params not found at {PARAMS_PATH}. Run run_sweep_abdlt_cat.py first."
        )

    print("=== Loading AAC cleaned cat data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"{len(df)} rows loaded\n")

    X = df[FEATURE_COLS]

    with open(PARAMS_PATH) as f:
        best_params = json.load(f)

    # --- outcome_type (classification) ---
    print("=== Final training: outcome_type (classification) ===")
    model = train_final_model(X, df[TARGET_CLF], best_params[TARGET_CLF])
    save_model(model, "abdlt_outcome_type_cat_tuned.pkl")

    # --- time_in_shelter (regression, trained on log scale) ---
    print("\n=== Final training: time_in_shelter (regression) ===")
    y_reg = np.log1p(df[TARGET_REG])
    model = train_final_model(X, y_reg, best_params[TARGET_REG])
    save_model(model, "abdlt_time_in_shelter_cat_tuned.pkl")

    # --- stay_category (classification) ---
    # print("\n=== Final training: stay_category (classification) ===")
    # model = train_final_model(X, df[TARGET_CLF2], best_params[TARGET_CLF2])
    # save_model(model, "abdlt_stay_category_cat_tuned.pkl")

    print("\nDone. All tuned cat models saved.")


if __name__ == "__main__":
    main()
