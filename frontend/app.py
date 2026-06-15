from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os

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


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    animal_type = data.get("animal_species", "dog").lower()

    features = {
        "age_intake":       data.get("age_intake"),
        "sex":              data.get("sex"),
        "spay_neuter":      data.get("spay_neuter"),
        "intake_month":     data.get("intake_month"),
        "intake_day":       data.get("intake_day"),
        "intake_year":      data.get("intake_year"),
        "animal_species":   animal_type,
        "colour":           data.get("colour"),
        "breed":            data.get("breed"),
        "intake_condition": data.get("intake_condition"),
        "intake_type":      data.get("intake_type"),
        "season":           get_season(data.get("intake_month", 1)),
        "animal_size": None, "is_mixed": None, "breed_1": None, "breed_2": None,
    }

    if animal_type == "dog":
        features.update({
            "good_with_children": None, "good_with_other_dogs": None,
            "shedding": None, "grooming": None, "drooling": None,
            "coat_length": None, "good_with_strangers": None,
            "playfulness": None, "protectiveness": None,
            "trainability": None, "energy": None, "barking": None,
        })
        time_model = dog_time_model
    else:
        features.update({
            "min_life_expectancy": None, "max_life_expectancy": None,
            "min_weight": None, "max_weight": None,
            "family_friendly": None, "shedding": None,
            "general_health": None, "playfulness": None,
            "children_friendly": None, "grooming": None,
            "intelligence": None, "other_pets_friendly": None,
        })
        time_model = cat_time_model

    input_df = pd.DataFrame([features])

    days_raw = time_model.predict(input_df)[0]
    days = round(float(np.expm1(days_raw)))

    return jsonify({
        "estimated_days_in_shelter": days
    })


if __name__ == "__main__":
    app.run(debug=True)
