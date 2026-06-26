from flask import Flask, request, jsonify, send_from_directory
import pickle
import subprocess
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(".")
from source.data_utils import categorize_dog_breed_by_size

app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble_part2", "models")

# Ensemble weights (optimized)
W_AAC_CAT = 0.61
W_AAC_DOG = 0.57

# Load short-term ensemble models
with open(os.path.join(MODELS_DIR, "model_cat_aac.pkl"), "rb") as f:
    cat_aac_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "model_cat_adb.pkl"), "rb") as f:
    cat_adb_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "model_dog_aac.pkl"), "rb") as f:
    dog_aac_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "model_dog_adb.pkl"), "rb") as f:
    dog_adb_model = pickle.load(f)

# Load long-term regression models (ABDLT, predict days in shelter)
with open(os.path.join(MODELS_DIR, "abdlt_time_in_shelter_cat_tuned.pkl"), "rb") as f:
    cat_abdlt_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, "ABDLT_time_in_shelter_dog_tuned.pkl"), "rb") as f:
    dog_abdlt_model = pickle.load(f)


FEATURE_COLS_CATS = [
    "age_intake", "sex", "spay_neuter", "intake_month", "intake_day", "intake_year",
    "animal_species", "colour", "intake_condition", "intake_type", "is_mixed",
    "breed_1", "breed_2", "min_life_expectancy", "max_life_expectancy",
    "min_weight", "max_weight", "family_friendly", "shedding", "general_health",
    "playfulness", "children_friendly", "grooming", "intelligence",
    "other_pets_friendly", "black", "white", "season", "population", "unemploy_rate",
]

FEATURE_COLS_DOGS = [
    "age_intake", "sex", "spay_neuter", "intake_month", "intake_day", "intake_year",
    "animal_species", "animal_size", "colour", "intake_condition", "intake_type",
    "is_mixed", "breed_1", "breed_2", "good_with_children", "good_with_other_dogs",
    "shedding", "grooming", "drooling", "coat_length", "good_with_strangers",
    "playfulness", "protectiveness", "trainability", "energy", "barking",
    "season", "population", "unemploy_rate",
]

# ABDLT long-term feature cols (no spay_neuter, no intake_condition)
ABDLT_FEATURE_COLS_DOGS = [
    "age_intake", "sex", "intake_month", "intake_day", "intake_year",
    "animal_species", "animal_size", "colour", "intake_type",
    "is_mixed", "breed_1", "breed_2", "good_with_children", "good_with_other_dogs",
    "shedding", "grooming", "drooling", "coat_length", "good_with_strangers",
    "playfulness", "protectiveness", "trainability", "energy", "barking",
    "season", "population", "unemploy_rate",
]

ABDLT_FEATURE_COLS_CATS = [
    "age_intake", "sex", "intake_month", "intake_day", "intake_year",
    "animal_species", "colour", "intake_type", "is_mixed",
    "breed_1", "breed_2", "family_friendly", "shedding", "general_health",
    "playfulness", "children_friendly", "grooming", "intelligence",
    "other_pets_friendly", "season", "black", "white", "population", "unemploy_rate",
]


def ensemble_predict_proba(p_aac, p_adb, w_aac):
    return w_aac * p_aac + (1 - w_aac) * p_adb


def get_season(month):
    month = int(month)
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    return "Winter"

def check_colour(row):
    '''
    Usage df[['black', 'white']] = df.apply(check_colour, axis=1, result_type='expand')
    '''
    if row['colour'] == 'Black':
        return 1, 0  
    elif row['colour'] == 'White':
        return 0, 1
    else:
        return 0, 0


FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/style.css')
def stylesheet():
    return send_from_directory(FRONTEND_DIR, 'style.css')


@app.route('/app.js')
def javascript():
    return send_from_directory(FRONTEND_DIR, 'app.js')


@app.route('/logo.png')
def logo():
    return send_from_directory(FRONTEND_DIR, 'IMG_0483.png')


PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)
CSV_PATH = os.path.join(FRONTEND_DIR, "single_prediction_prompt.csv")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    animal_type = data.get("animal_species", "dog").lower()

    # --- 1. Build the initial row with user inputs, Gemini fields left blank ---
    row = {
        "animal_type":      animal_type,
        "city":             data.get("city"),
        "age_intake":       data.get("age_intake"),
        "sex":              data.get("sex"),
        "spay_neuter":      data.get("spay_neuter"),
        "intake_month":     data.get("intake_month"),
        "intake_day":       data.get("intake_day"),
        "intake_year":      data.get("intake_year"),
        "animal_species":   animal_type,
        "colour":           data.get("colour"),
        "intake_condition": data.get("intake_condition"),
        "intake_type":      data.get("intake_type"),
        "season":           get_season(data.get("intake_month", 1)),
        "is_mixed":         data.get("is_mixed"),
        "breed_1":          data.get("breed_1"),
        "breed_2":          data.get("breed_2") or "not given",
    }

    if animal_type == "dog":
        row.update({
            "animal_size":          data.get("size"),
            "good_with_children":   None, "good_with_other_dogs": None,
            "shedding":             None, "grooming":             None,
            "drooling":             None, "coat_length":          None,
            "good_with_strangers":  None, "playfulness":          None,
            "protectiveness":       None, "trainability":         None,
            "energy":               None, "barking":              None,
            "population":           None, "unemploy_rate":        None,
        })
    else:
        black, white = check_colour(row)
        row.update({
            "min_life_expectancy":  None, "max_life_expectancy":  None,
            "min_weight":           None, "max_weight":           None,
            "family_friendly":      None, "shedding":             None,
            "general_health":       None, "playfulness":          None,
            "children_friendly":    None, "grooming":             None,
            "intelligence":         None, "other_pets_friendly":  None,
            "black":                black, "white":               white,
            "population":           None, "unemploy_rate":        None,
        })

    if row["breed_2"]in ["None", None, "N/A","N"]:
        row["breed_2"] = "not given"

    print(row["breed_2"])


    # --- 2. Save to CSV so the prompting script can read it ---
    pd.DataFrame([row]).to_csv(CSV_PATH, index=False)

    # --- 3. Run Gemini prompting script to fill in breed trait columns ---
    try:
        subprocess.run(
            [sys.executable, "source/prompting_template_aac.py"],
            cwd=PROJECT_ROOT,
            check=True
        )
    except subprocess.CalledProcessError:
        return jsonify({"error": "high_demand"}), 503

    # --- 4. Read the completed CSV back ---
    completed = pd.read_csv(CSV_PATH)

    # --- 5. Ensemble predict ---
    if animal_type == "dog":
        input_df = completed[FEATURE_COLS_DOGS]
        p_aac = dog_aac_model.predict_proba(input_df)
        p_adb = dog_adb_model.predict_proba(input_df)
        blended = ensemble_predict_proba(p_aac, p_adb, W_AAC_DOG)
        classes = dog_aac_model.classes_
    else:
        input_df = completed[FEATURE_COLS_CATS]
        p_aac = cat_aac_model.predict_proba(input_df)
        p_adb = cat_adb_model.predict_proba(input_df)
        blended = ensemble_predict_proba(p_aac, p_adb, W_AAC_CAT)
        classes = cat_aac_model.classes_

    proba_list = blended[0].tolist()
    predicted_class = classes[int(np.argmax(blended[0]))]
    print(dict(zip(classes.tolist(), proba_list)), "->", predicted_class)

    # --- 6. Long-term regression prediction (ABDLT) ---
    if animal_type == "dog":
        abdlt_input = completed[ABDLT_FEATURE_COLS_DOGS].copy()
        longterm_days = float(np.expm1(dog_abdlt_model.predict(abdlt_input)[0]))
    else:
        abdlt_input = completed[ABDLT_FEATURE_COLS_CATS].copy()
        # breed_2 was all-NaN (float) in cat training data — replace string sentinels
        if "breed_2" in abdlt_input.columns:
            abdlt_input["breed_2"] = pd.to_numeric(abdlt_input["breed_2"], errors="coerce")
        longterm_days = float(np.expm1(cat_abdlt_model.predict(abdlt_input)[0]))

    print(f"Long-term prediction: {longterm_days:.1f} days")

    return jsonify({
        "predicted_bin": predicted_class,
        "probabilities": dict(zip(classes.tolist(), proba_list)),
        "longterm_days": round(longterm_days, 1),
    })


if __name__ == "__main__":
    app.run(debug=True)
