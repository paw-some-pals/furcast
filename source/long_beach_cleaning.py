
#imports
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from data_utils import simplify_animal_species


#only keeping cat and dog using function from utils 
def keep_cat_dog(df):
    df.rename(columns={'Animal Type': 'animal_species'}, inplace=True) # make consistent with other datasets
    df['animal_species'] = df['animal_species'].str.lower()
    new_df = simplify_animal_species(df)  
    return new_df   


def read_file(path):
    lb_df = pd.read_csv("/Users/heerperchani/Desktop/AI4Good/furcast/datasets/animal-shelter-intakes-and-outcomes.csv")
    return lb_df

def clean_data(df):
    # removing the columns we will not be using
    df = df.drop([
        "Kennel ID", "latitude", "longitude", "Animal Name", "Secondary Color",
        "Intake Subtype", "Reason for Intake", "Crossing", "Jurisdiction",
        "Outcome Subtype", "outcome_is_dead", "was_outcome_alive", "geopoint",
        "is_current_month", "intake_duration", "outcome_is_current",
        "outcome_is_other", "outcome_is_alive", "intake_is_dead"
    ], axis=1)

    # age added in years float
    age_in_days = pd.to_datetime(df["Intake Date"]) - pd.to_datetime(df["DOB"])
    df["age_in_years"] = (age_in_days / pd.to_timedelta(365.4525, "D")).round(2)
    

    # number of days spent in shelter added float
    days_spent = pd.to_datetime(df["Outcome Date"]) - pd.to_datetime(df["Intake Date"])
    df["duration_in_days"] = days_spent / pd.to_timedelta(1, "D")
    

    # tried looking for duplicates but no rows were dropped
    df.drop_duplicates()

    df[df["age_in_years"].isna()]
    # there are 6968 rows with no DOB so no age

    return df 

def main():

    my_df = read_file("/Users/heerperchani/Desktop/AI4Good/furcast/datasets/animal-shelter-intakes-and-outcomes.csv")
    cleaned_df = clean_data(my_df)
    cleaned_df = spay_neuter(cleaned_df)
    cleaned_df = simplifying_intake_condition(cleaned_df)
    cleaned_df = simplifying_intake_type(cleaned_df)
    cleaned_df = keep_cat_dog(cleaned_df)
    print(cleaned_df["animal_species"].value_counts())


def spay_neuter(df):
    for index in df.index:
        sex = df.at[index, "Sex"]
        if sex == "Male":
            df.at[index, "Sex"] = "Male"
            df.at[index, "spay_neuter"] = "No"
        elif sex == "Female":
            df.at[index, "Sex"] = "Female"
            df.at[index, "spay_neuter"] = "No"
        elif sex == "Neutered":
            df.at[index, "Sex"] = "Male"
            df.at[index, "spay_neuter"] = "Yes"
        elif sex == "Spayed":
            df.at[index, "Sex"] = "Female"
            df.at[index, "spay_neuter"] = "Yes"
        else:
            df.at[index, "Sex"] = "Unknown"
            df.at[index, "spay_neuter"] = "Unknown"

    return df



# # all ill mapped to sick
# # underage/weight mapped to sick
# # everything else is other 
def simplifying_intake_condition(df):
    for index in df.index:
        condition = df.at[index, "Intake Condition"]
        if condition == "NORMAL":
            df.at[index, "Intake Condition"] = "Normal"
        elif "INJURED" in condition:
            df.at[index, "Intake Condition"] = "Injured"
        elif "ILL" in condition or condition == "UNDER AGE/WEIGHT":
            df.at[index, "Intake Condition"] = "Sick"
        elif condition == "FERAL":
            df.at[index, "Intake Condition"] = "Feral"
        elif condition == "AGED":
            df.at[index, "Intake Condition"] = "Aged"
        else:
            df.at[index, "Intake Condition"] = "Other"
    return df 

# simplifying intake type
def simplifying_intake_type(df):
    for index in df.index:
        type = df.at[index, "Intake Type"]
        if type == "STRAY":
            df.at[index, "Intake Type"] = "Stray"
        elif type in ["OWNER SURRENDER"]:
            df.at[index, "Intake Type"] = "Owner Surrender"
        elif type in ["OWNER SURRENDER"]:
            df.at[index, "Intake Type"] = "Euthanasia Request"
        elif type in ["WILDLIFE", "WELFARE SEIZED", "CONFISCATE", "RETURN", "QUARANTINE", "SAFE KEEP", "TRAP, NEUTER, RETURN", "FOSTER", "TRANSFER", "Adopted Animal Return"]:
            df.at[index, "Intake Type"] = "Other"

    return df


main()



