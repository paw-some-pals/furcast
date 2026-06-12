import sys
import os
import pickle


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

AAC_cat = pd.read_csv("")
ADB_cat = pd.read_csv("") 

AAC_dog = pd.read_csv("")
ADB_dog = pd.read_csv("")

FEATURE_COLS_DOGS = [
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
    "season"
]

FEATURE_COLS_CATS = [
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

TARGET_CLF = "outcome_type"     # Category: Adoption, Transfer, Euthanasia, etc.
TARGET_REG = "time_in_shelter"


def make_split_sets(feature_cols, df1,df2,testsize = 0.15, randomstate=42,cat):
    

    df1_X = df1[feature_cols] #pass the austin one
    df2_X = df2[feature_cols] #pass the ADB one

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(df1_X, df1_y, test_size=testsize, random_state=randomstate)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(df2_X, df2_y, test_size=testsize, random_state=randomstate)


    #make a copy of X_test_1, y_test_1 and delete spay and neuter 
    #make a copy of X_test_2, y_test_2 and add a col called spay neuter with Unknown and intake conditon with Other

    copy_X_test_1 = X_test_1.copy()
    copy_y_test_1 = y_test_1.copy()

    copy_X_test_1.remove(columns=["spay_neuter","intake_condition"])



    copy_X_test_2 = X_test_2.copy()
    copy_y_test_2 = y_test_2.copy()

    copy_X_test_2["spay_neuter"] = "Unknown"
    copy_X_test_2["intake_condition"] = "Other"


    #mix x1 copy with x2 and mix x2 copy with x1 
    if cat ==1:
        ABD_cat_test = pd.concat([copy_X_test_1, X_test_2], ignore_index=True)  #this is the one without spay neuter for ABD
        AAC_cat_test = pd.concat([X_test_1, copy_X_test_2], ignore_index=True)  #this is the one with spay neuter for ACC 
    else:
        ABD_dog_test = pd.concat([copy_X_test_1, X_test_2], ignore_index=True)  #this is the one without spay neuter for ABD
        AAC_dog_test = pd.concat([X_test_1, copy_X_test_2], ignore_index=True)  #this is the one with spay neuter for ACC


    #SAVING FINAL DATA SETS 
    ABD_cat_test.to_csv("datasets/FINAL/ABD_MODEL_CAT_TEST.csv", index=False)
    ABD_dog_test.to_csv("datasets/FINAL/ABD_MODEL_DOG_TEST.csv", index=False)
    AAC_cat_test.to_csv("datasets/FINAL/AAC_MODEL_CAT_TEST.csv", index=False)
    AAC_dog_test.to_csv("datasets/FINAL/AAC_MODEL_DOG_TEST.csv", index=False)


