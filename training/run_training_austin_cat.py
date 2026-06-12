"""
Run training pipeline for the AAC (Austin Animal Center) dataset.

Assumes aac_cleaning.py has already been run and datasets/aac_cleaned.csv exists.

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
from sklearn.model_selection import cross_val_score, train_test_split

# Allow importing from this same folder (training/)
sys.path.insert(0, os.path.dirname(__file__))

from random_forest_training import build_random_forest_pipeline, train_random_forest
from evaluate import evaluate

# Paths to the cleaned input data and the folder where trained models will be saved
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "acc_cat_final.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# The columns the model uses as inputs — these are facts known about an animal at intake
FEATURE_COLS = [
    "age_intake",       # Age of the animal when it arrived
    "sex",              # Male or female
    "spay_neuter",      # Whether the animal is spayed/neutered
    "intake_month",     # Month of arrival (1–12)
    "intake_day",       # Day of arrival (1–31)
    "intake_year",      # Year of arrival
    "animal_species",   # dog or cat
    "animal_size",      
    "colour",           # colour
    "breed",            # Breed of the animal
    "intake_condition", # Health condition at arrival (e.g. Normal, Injured)
    "intake_type",      # How the animal arrived (e.g. Stray, Owner Surrender)
    "is_mixed",
    "breed_1",
    "breed_2",
    "min_life_expectancy",
    "max_life_expectancy",
    "min_weight",
    "max_weight",
    "family_friendly",
    "shedding",
    "general_health",
    "playfulness",
    "children_friendly",
    "grooming",
    "intelligence",
    "other_pets_friendly",
    "season"
]

# targets
TARGET_CLF = "outcome_type"     # Category: Adoption, Transfer, Euthanasia, etc.
TARGET_REG = "time_in_shelter"  # Number of days the animal stayed in the shelter


def save_model(model, filename):
    """Serialize the trained model to disk so it can be loaded and used later."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    print("=== Loading AAC cleaned cat data ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"{len(df)} rows loaded\n")

    # select features
    X = df[FEATURE_COLS]

    # --- Model 1: Predict what outcome an animal will have (classification) ---
    print("=== Training: outcome_type (classification) ===")
    y_clf = df[TARGET_CLF]

    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf), X, y_clf, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Split data: 80% train and 20% test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)

    # Print accuracy metrics on the held-out test set
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "aac_outcome_type_cat.pkl")

    # --- Model 2: Predict how many days an animal will stay (regression) ---

    print("\n=== Training: time_in_shelter cat(regression) ===")

    # log1p compresses the range of day values so extreme outliers (e.g. 500-day stays) 
    # don't dominate training. Learn the model learns on a more balanced scale (len of stay is right skewed) ( can rethink/discuss in TA session)
    # https://medium.com/@noorfatimaafzalbutt/understanding-np-log-and-np-log1p-in-numpy-99cefa89cd30#:~:text=transformation%20of%20skewed%20data%20(like%20when%20dealing%20with%20highly%20skewed%20distributions%20in%20data%20preprocessing)%20is%20necessary.
    
    y_reg = np.log1p(df[TARGET_REG])

    cv_mae = -cross_val_score(build_random_forest_pipeline(X, y_reg), X, y_reg, cv=5, scoring="neg_mean_absolute_error")
    print(f"CV MAE (log scale): {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    model_reg = train_random_forest(X_train, y_train)

    # expm1 reverses the log transform so error metrics are back in actual days for easier analysis
    y_pred = np.expm1(model_reg.predict(X_test))
    y_test_orig = np.expm1(y_test)
    
    # MAE: average prediction error in days; RMSE: penalizes large errors more heavily
    print(f"MAE:  {mean_absolute_error(y_test_orig, y_pred):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test_orig, y_pred):.2f}")
    save_model(model_reg, "aac_time_in_shelter_cat.pkl")


if __name__ == "__main__":
    main()
