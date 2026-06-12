import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from source.plotting import mutual_info_regression_matrix

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET = 'datasets/final_df_aac_dogs.csv'
OUTPUT   = 'figures/mi_heatmap_dog_time_in_shelter.png'
TARGET   = 'time_in_shelter'   # rows missing this value are dropped; set to None to skip

FEATURE_COLS = [
    'age_intake',
    'sex',
    'spay_neuter',
    #'intake_month',
    #'intake_day',
    #'intake_year',
    'animal_species',
    'animal_size',
    'colour',
    #'breed',
    'intake_condition',
    'intake_type',
    #'is_mixed',
    'breed_1',
    #'breed_2',
    #'good_with_children',
    #'good_with_other_dogs',
    'shedding',
    'grooming',
    #'drooling',
    #'coat_length',
    'good_with_strangers',
    'playfulness',
    'protectiveness',
    'trainability',
    'energy',
    'barking',
    #'stay_category',
    'season',
    'population',
    #'unemploy',
    'time_in_shelter',
    #'outcome_type',
    
]

# ── END CONFIG ────────────────────────────────────────────────────────────────

df = pd.read_csv(DATASET)

if TARGET:
    df = df.dropna(subset=[TARGET])

df = df[FEATURE_COLS]

print(f"Shape: {df.shape}")
print(df.dtypes)

mutual_info_regression_matrix(df, filename=OUTPUT)

print(f"Saved to {OUTPUT}")
