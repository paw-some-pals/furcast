import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from source.plotting import mutual_info_regression_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET     = 'datasets/final_df_aac_dogs.csv'
TARGET      = 'time_in_shelter'   # always appended to every variation
OUTPUT_DIR  = 'figures/mi_heatmaps_dog_aac'  # where to save the heatmaps

# Constraints enforced in variation design:
#   - month and season never together
#   - breed and breed_1 never together
#   - breed never appears alongside breed-info trait columns
VARIATIONS = [
    {
        'name': 'v1_intake_timing',
        'features': [
            'age_intake', 'sex', 'spay_neuter', 'intake_month', 'intake_day',
            'intake_year', 'animal_species', 'animal_size', 'intake_condition',
            'intake_type',
        ],
    },
    {
        'name': 'v2_breed_season',   # breed (no breed_1, no breed info), season (no month)
        'features': [
            'breed', 'is_mixed', 'animal_size', 'animal_species', 'colour',
            'intake_condition', 'season', 'population', 'age_intake',
        ],
    },
    {
        'name': 'v3_breed1_month',   # breed_1/breed_2 (no breed), month (no season)
        'features': [
            'breed_1', 'breed_2', 'is_mixed', 'animal_size', 'age_intake',
            'sex', 'spay_neuter', 'intake_month', 'intake_condition', 'population',
        ],
    },
    {
        'name': 'v4_behavioral_traits',   # no breed, no breed_1
        'features': [
            'good_with_children', 'good_with_other_dogs', 'good_with_strangers',
            'playfulness', 'protectiveness', 'trainability', 'energy', 'barking',
            'age_intake', 'animal_size',
        ],
    },
    {
        'name': 'v5_physical_traits',   # no breed, no breed_1
        'features': [
            'shedding', 'grooming', 'drooling', 'coat_length', 'is_mixed',
            'animal_size', 'animal_species', 'age_intake', 'sex',
        ],
    },
    {
        'name': 'v6_season_context',   # season (no month), no breed
        'features': [
            'season', 'population', 'animal_size', 'animal_species', 'intake_condition',
            'intake_type', 'is_mixed', 'spay_neuter', 'sex',
        ],
    },
    {
        'name': 'v7_colour_breed1',   # breed_1 (no breed), no month, no season
        'features': [
            'colour', 'breed_1', 'breed_2', 'is_mixed', 'animal_species', 'animal_size',
            'intake_type', 'intake_condition', 'population', 'intake_year',
        ],
    },
    {
        'name': 'v8_traits_context',   # mixed physical traits with demographics, no breed
        'features': [
            'good_with_children', 'good_with_other_dogs', 'shedding', 'grooming',
            'drooling', 'coat_length', 'is_mixed', 'animal_size', 'population', 'sex',
        ],
    },
    {
        'name': 'v9_social_traits_season',   # social/energy traits, season (no month), no breed
        'features': [
            'good_with_strangers', 'playfulness', 'protectiveness', 'trainability',
            'energy', 'barking', 'age_intake', 'sex', 'intake_condition', 'season',
        ],
    },
    {
        'name': 'v10_overview',   # broad overview, season (no month), no breed columns
        'features': [
            'age_intake', 'sex', 'spay_neuter', 'animal_size', 'intake_condition',
            'season', 'population', 'is_mixed', 'colour',
        ],
    },
]
# ── END CONFIG ────────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATASET)
df_raw = df_raw.dropna(subset=[TARGET])

os.makedirs(OUTPUT_DIR, exist_ok=True)

for var in VARIATIONS:
    cols = var['features'] + [TARGET]
    missing = [c for c in cols if c not in df_raw.columns]
    if missing:
        print(f"[{var['name']}] skipping — missing columns: {missing}")
        continue

    df_var = df_raw[cols].copy()
    output = os.path.join(OUTPUT_DIR, f"mi_heatmap_{var['name']}.png")

    print(f"[{var['name']}] shape={df_var.shape}  →  {output}")
    mutual_info_regression_matrix(df_var, filename=output)
    print(f"[{var['name']}] saved.")
