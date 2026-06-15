"""
Run training pipeline for the Adoptions by breed and date dataset.

Assumes cleanadoptionsbybreedanddate.py has already been run and datasets/dogshorttermABD.csv exists.

Trains a Random Forest for both:
  - outcome_type (classification)
  - time_in_shelter (regression)

Saves trained models to models/aac_outcome_type.pkl and models/aac_time_in_shelter.pkl.
"""

import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Allow importing from this same folder (training/)
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # training/ for shared modules

from random_forest_training import train_random_forest
from evaluate import evaluate

# Paths to the cleaned input data and the folder where trained models will be saved
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "ABDST_output_dog_pop.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# The columns the model uses as inputs — these are facts known about an animal at intake
FEATURE_COLS = [ # MI with time_in_shelter
    "age_intake",           # 0.159
    "sex",                  # 0.014
    #"spay_neuter",          # 0.069
    "intake_month",         # 0.079
    "intake_day",
    "intake_year",          # 0.049
    "animal_species",
    #"animal_size",          # 0.058
    "colour",               # 0.016
    "breed",                # 0.154
    #"intake_condition",     # 0.022
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

def save_model(model, filename):
    """Serialize the trained model to disk so it can be loaded and used later."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    print("=== Loading shorttermABD cleaned data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"{len(df)} rows loaded\n")

    # select features
    X = df[FEATURE_COLS]

    # --- Model 1: Predict what outcome an animal will have (classification) ---
    print("=== Training: outcome_type (classification) ===")
    y_clf = df[TARGET_CLF]

    # Split data: 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)

    # Print accuracy metrics on the held-out test set
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "dogshorttermABD.pkl")

    # --- Model 2: Predict how many days an animal will stay (regression) ---

    print("\n=== Training: time_in_shelter (regression) ===")

    # log1p compresses the range of day values so extreme outliers (e.g. 500-day stays) 
    # don't dominate training. Learn the model learns on a more balanced scale (len of stay is right skewed) ( can rethink/discuss in TA session)
    # https://medium.com/@noorfatimaafzalbutt/understanding-np-log-and-np-log1p-in-numpy-99cefa89cd30#:~:text=transformation%20of%20skewed%20data%20(like%20when%20dealing%20with%20highly%20skewed%20distributions%20in%20data%20preprocessing)%20is%20necessary.
    
    y_reg = np.log1p(df[TARGET_REG])
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    model_reg = train_random_forest(X_train, y_train)

    # expm1 reverses the log transform so error metrics are back in actual days for easier analysis
    y_pred = np.expm1(model_reg.predict(X_test))
    y_test_orig = np.expm1(y_test)
    
    # MAE: average prediction error in days; RMSE: penalizes large errors more heavily
    print(f"MAE:  {mean_absolute_error(y_test_orig, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test_orig, y_pred):.2f}")
    save_model(model_reg, "catshorttermabdmodel.pkl")


if __name__ == "__main__":
    main()
