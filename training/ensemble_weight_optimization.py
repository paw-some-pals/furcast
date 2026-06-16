import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

cat_val = pd.read_csv("datasets/ensemble_part2/combined_cat_val.csv")
dog_val = pd.read_csv("datasets/ensemble_part2/combined_dog_val.csv")
cat_test = pd.read_csv("datasets/ensemble_part2/combined_cat_test.csv")
dog_test = pd.read_csv("datasets/ensemble_part2/combined_dog_test.csv")


with open("models/ensemble_part2/models/model_cat_aac.pkl", "rb") as f:
    aac_cat_model = pickle.load(f)

with open("models/ensemble_part2/models/model_cat_adb.pkl", "rb") as f:
    adb_cat_model = pickle.load(f)

with open("models/ensemble_part2/models/model_dog_aac.pkl", "rb") as f:
    aac_dog_model = pickle.load(f)

with open("models/ensemble_part2/models/model_dog_adb.pkl", "rb") as f:
    adb_dog_model = pickle.load(f)

FEATURE_COLS_DOGS_FULL = [
    "age_intake",       # Age of the animal when it arrived
    "sex",              # Male or female
    "spay_neuter",      # Whether the animal is spayed/neutered
    "intake_month",     # Month of arrival (1–12)
    "intake_day",       # Day of arrival (1–31)
    "intake_year",      # Year of arrival
    "animal_species",   # dog or cat
    "animal_size",      
    "colour",           # colour        
    "intake_condition", # Health condition at arrival (e.g. Normal, Injured)
    "intake_type",      # How the animal arrived (e.g. Stray, Owner Surrender)
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
    "unemploy_rate"
]

FEATURE_COLS_CATS_FULL = [
    "age_intake",       # Age of the animal when it arrived
    "sex",              # Male or female
    "spay_neuter",      # Whether the animal is spayed/neutered
    "intake_month",     # Month of arrival (1–12)
    "intake_day",       # Day of arrival (1–31)
    "intake_year",      # Year of arrival
    "animal_species",   # dog or cat
    "colour",           # colour
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
    "black",
    "white",
    "season",
    "population",
    "unemploy_rate"
]

# FEATURE_COLS_DOGS_RESTRICTED = [f for f in FEATURE_COLS_DOGS_FULL if f not in ("spay_neuter", "intake_condition")]
# FEATURE_COLS_CATS_RESTRICTED = [f for f in FEATURE_COLS_CATS_FULL if f not in ("spay_neuter", "intake_condition")]


TARGET_CLF = "stay_category"







def find_best_weight(y_true, p_aac, p_adb):
    best_w = 0
    best_loss = float("inf")

    for w in np.linspace(0, 1, 101):
        p = w * p_aac + (1 - w) * p_adb
        loss = log_loss(y_true, p)

        if loss < best_loss:
            best_loss = loss
            best_w = w

    return best_w, best_loss


# --- Cat Full ---
cat_val_X_full = cat_val[FEATURE_COLS_CATS_FULL].copy()
cat_val_X_full["breed_2"] = cat_val_X_full["breed_2"].fillna("N/A")
cat_val_y = cat_val[TARGET_CLF]

p_aac_cat_full = aac_cat_model.predict_proba(cat_val_X_full)
p_adb_cat_full = adb_cat_model.predict_proba(cat_val_X_full)

best_w_cat_full, best_loss_cat_full = find_best_weight(cat_val_y, p_aac_cat_full, p_adb_cat_full)
print(f"Cat Full  — AAC weight: {best_w_cat_full:.2f}, ADB weight: {1 - best_w_cat_full:.2f}, val log-loss: {best_loss_cat_full:.4f}")

# --- Dog Full ---
dog_val_X_full = dog_val[FEATURE_COLS_DOGS_FULL].copy()
dog_val_X_full["breed_2"] = dog_val_X_full["breed_2"].fillna("N/A")
dog_val_y = dog_val[TARGET_CLF]

p_aac_dog_full = aac_dog_model.predict_proba(dog_val_X_full)
p_adb_dog_full = adb_dog_model.predict_proba(dog_val_X_full)

best_w_dog_full, best_loss_dog_full = find_best_weight(dog_val_y, p_aac_dog_full, p_adb_dog_full)
print(f"Dog Full  — AAC weight: {best_w_dog_full:.2f}, ADB weight: {1 - best_w_dog_full:.2f}, val log-loss: {best_loss_dog_full:.4f}")



