"""
Training on merged AAC + ABST data for cats and dogs.

Loads final_df_aac_cats/dogs.csv and ABST_output_cat/dog_pop.csv,
concatenates them, then trains with 5-fold CV + one held-out run.

Cat: drops spay_neuter, intake_condition, animal_size before training.
Dog: drops spay_neuter, intake_condition before training.
-----------------
Update: missingness indicators added for spay_neuter and intake_condition, and missing values filled with "Unknown"/"Other" to allow model to learn from missingness patterns.

Feature columns are filtered to those present in the merged dataframe
so the file still runs with different feature sets in the dfs
"""

import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import (
    cross_val_score,
    cross_val_predict,
    train_test_split,
    KFold,
    StratifiedKFold,
)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from random_forest_training import build_random_forest_pipeline, train_random_forest
from evaluate import evaluate

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "models")

TARGET_CLF  = "outcome_type"
TARGET_REG  = "time_in_shelter"
TARGET_CLF2 = "stay_category"

FEATURE_COLS_CAT = [
    "age_intake",
    "sex",
    "spay_neuter",
    "spay_neuter_missing",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    # animal_size dropped
    "colour",
    "breed",
    "intake_condition",
    "intake_condition_missing",
    "intake_type",
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
    "season",
    "black",
    "white",
    "population",
    "unemploy_rate",
]

FEATURE_COLS_DOG = [
    "age_intake",
    "sex",
    "spay_neuter",
    "spay_neuter_missing",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    "animal_size",
    "colour",
    "breed",
    "intake_condition",
    "intake_condition_missing",
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


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def run_cat():
    print("=== Loading and merging cat data ===")
    aac  = pd.read_csv(os.path.join(DATASETS_DIR, "final_df_aac_cats.csv"))
    abst = pd.read_csv(os.path.join(DATASETS_DIR, "ABDST_output_cat_pop.csv"))
    df   = pd.concat([aac, abst], ignore_index=True)
    print(f"{len(df)} rows after merge ({len(aac)} AAC + {len(abst)} ABST)\n")

    # missingness indication 
    df["spay_neuter_missing"]      = df["spay_neuter"].isna().astype(int)
    df["intake_condition_missing"] = df["intake_condition"].isna().astype(int)
    df["spay_neuter"]              = df["spay_neuter"].fillna("Unknown")
    df["intake_condition"]         = df["intake_condition"].fillna("Other")

    available = [c for c in FEATURE_COLS_CAT if c in df.columns]
    X = df[available]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Training: outcome_type (classification) cat ===")
    y_clf = df[TARGET_CLF]
    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf), X, y_clf, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "aac_abst_merged_outcome_type_cat.pkl")
    """
    print("\n=== Training: time_in_shelter cat (regression) ===")
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
    save_model(model_reg, "aac_abst_merged_time_in_shelter_cat.pkl")
    """
    print("\n=== Training: stay_category (classification) cat ===")
    y_clf2 = df[TARGET_CLF2]
    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf2), X, y_clf2, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf2, test_size=0.2, random_state=42)
    model_clf2 = train_random_forest(X_train, y_train)
    evaluate(model_clf2, X_test, y_test)
    save_model(model_clf2, "aac_abst_merged_stay_category_cat.pkl")


def run_dog():
    print("=== Loading and merging dog data ===")
    aac  = pd.read_csv(os.path.join(DATASETS_DIR, "final_df_aac_dogs.csv"))
    abst = pd.read_csv(os.path.join(DATASETS_DIR, "ABDST_output_dog_pop.csv"))
    df   = pd.concat([aac, abst], ignore_index=True)
    print(f"{len(df)} rows after merge ({len(aac)} AAC + {len(abst)} ABST)\n")

    # missingness indication
    df["spay_neuter_missing"]      = df["spay_neuter"].isna().astype(int)
    df["intake_condition_missing"] = df["intake_condition"].isna().astype(int)
    df["spay_neuter"]              = df["spay_neuter"].fillna("Unknown")
    df["intake_condition"]         = df["intake_condition"].fillna("Other")

    available = [c for c in FEATURE_COLS_DOG if c in df.columns]
    X = df[available]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # manual to ensure shuffle=True 
    kf  = KFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Training: outcome_type (classification) dog ===")
    y_clf = df[TARGET_CLF]
    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf), X, y_clf, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    model_clf = train_random_forest(X_train, y_train)
    evaluate(model_clf, X_test, y_test)
    save_model(model_clf, "aac_abst_merged_outcome_type_dog.pkl")
    
    # regression commented out, likely will not use, takes a while to run
    # for cat and dog in these runs, no notable increase in metrics
    """ 

    print("\n=== Training: time_in_shelter dog (regression) ===")
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
    save_model(model_reg, "aac_abst_merged_time_in_shelter_dog.pkl")
    """
    print("\n=== Training: stay_category (classification) dog ===")
    y_clf2 = df[TARGET_CLF2]
    cv_scores = cross_val_score(build_random_forest_pipeline(X, y_clf2), X, y_clf2, cv=skf, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf2, test_size=0.2, random_state=42)
    model_clf2 = train_random_forest(X_train, y_train)
    evaluate(model_clf2, X_test, y_test)
    save_model(model_clf2, "aac_abst_merged_stay_category_dog.pkl")


if __name__ == "__main__":
    run_cat()
    print("\n" + "=" * 60 + "\n") # divider between cat and dog results (easier to read results in term)
    run_dog()
