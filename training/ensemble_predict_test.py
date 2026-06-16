import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cat_test = pd.read_csv("datasets/final_ensemble/combined_cat_test.csv")
dog_test = pd.read_csv("datasets/final_ensemble/combined_dog_test.csv")

with open("models/ensemble_part2/models/model_cat_aac.pkl", "rb") as f:
    aac_cat_model = pickle.load(f)
with open("models/ensemble_part2/models/model_cat_adb.pkl", "rb") as f:
    adb_cat_model = pickle.load(f)
with open("models/ensemble_part2/models/model_dog_aac.pkl", "rb") as f:
    aac_dog_model = pickle.load(f)
with open("models/ensemble_part2/models/model_dog_adb.pkl", "rb") as f:
    adb_dog_model = pickle.load(f)

FEATURE_COLS_CATS_FULL = [
    "age_intake", "sex", "spay_neuter", "intake_month", "intake_day", "intake_year",
    "animal_species", "colour", "intake_condition", "intake_type", "is_mixed",
    "breed_1", "breed_2", "min_life_expectancy", "max_life_expectancy",
    "min_weight", "max_weight", "family_friendly", "shedding", "general_health",
    "playfulness", "children_friendly", "grooming", "intelligence",
    "other_pets_friendly", "black", "white", "season", "population", "unemploy_rate"
]

FEATURE_COLS_DOGS_FULL = [
    "age_intake", "sex", "spay_neuter", "intake_month", "intake_day", "intake_year",
    "animal_species", "animal_size", "colour", "intake_condition", "intake_type",
    "is_mixed", "breed_1", "breed_2", "good_with_children", "good_with_other_dogs",
    "shedding", "grooming", "drooling", "coat_length", "good_with_strangers",
    "playfulness", "protectiveness", "trainability", "energy", "barking",
    "season", "population", "unemploy_rate"
]

TARGET_CLF = "stay_category"

# Weights from ensemble_train.py — update these after running weight optimization
W_AAC_CAT = 0.61
W_AAC_DOG = 0.57


def ensemble_predict(p_aac, p_adb, w_aac, classes):
    p_combined = w_aac * p_aac + (1 - w_aac) * p_adb
    indices = p_combined.argmax(axis=1)
    return classes[indices]


def evaluate(name, y_true, y_pred, classes):
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=classes))


# --- Cat ---
cat_test_X = cat_test[FEATURE_COLS_CATS_FULL]
cat_test_y = cat_test[TARGET_CLF]

p_aac_cat = aac_cat_model.predict_proba(cat_test_X)
p_adb_cat = adb_cat_model.predict_proba(cat_test_X)

classes_cat = aac_cat_model.classes_
cat_preds = ensemble_predict(p_aac_cat, p_adb_cat, W_AAC_CAT, classes_cat)
evaluate("Cat Ensemble", cat_test_y, cat_preds, classes_cat)

# --- Dog ---
dog_test_X = dog_test[FEATURE_COLS_DOGS_FULL]
dog_test_y = dog_test[TARGET_CLF]

p_aac_dog = aac_dog_model.predict_proba(dog_test_X)

p_adb_dog = adb_dog_model.predict_proba(dog_test_X)

classes_dog = aac_dog_model.classes_
dog_preds = ensemble_predict(p_aac_dog, p_adb_dog, W_AAC_DOG, classes_dog)
evaluate("Dog Ensemble", dog_test_y, dog_preds, classes_dog)
