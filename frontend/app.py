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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Load models
with open(os.path.join(MODELS_DIR, "aac_time_in_shelter_dog.pkl"), "rb") as f:
    dog_time_model = pickle.load(f)

with open(os.path.join(MODELS_DIR, "aac_time_in_shelter_cat.pkl"), "rb") as f:
    cat_time_model = pickle.load(f)


def get_season(month):
    month = int(month)
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    return "Winter"


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
        "breed_2":          data.get("breed_2"),
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
        time_model = dog_time_model
    else:
        row.update({
            "min_life_expectancy":  None, "max_life_expectancy":  None,
            "min_weight":           None, "max_weight":           None,
            "family_friendly":      None, "shedding":             None,
            "general_health":       None, "playfulness":          None,
            "children_friendly":    None, "grooming":             None,
            "intelligence":         None, "other_pets_friendly":  None,
            "population":           None, "unemploy_rate":        None,
        })
        time_model = cat_time_model

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

    # --- 5. Drop metadata columns the model doesn't use ---
    # drop_cols = ["animal_type", "city"]
    # input_df = completed.drop(columns=[c for c in drop_cols if c in completed.columns])

    # --- 6. Predict and return ---
    completed["breed"] = completed["breed_1"]
    completed["animal_size"] = "Unknown"
    input_df = completed
    days_raw = time_model.predict_proba(input_df)[0]
    days = round(float(np.expm1(days_raw)))

    return jsonify({"estimated_days_in_shelter": days})


if __name__ == "__main__":
    app.run(debug=True)
