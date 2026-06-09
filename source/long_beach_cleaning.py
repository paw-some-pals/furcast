
#imports
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from data_utils import simplify_animal_species, simplifying_intake_type


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




def main():

    my_df = read_file("/Users/heerperchani/Desktop/AI4Good/furcast/datasets/animal-shelter-intakes-and-outcomes.csv")
    cleaned_df = clean_data(my_df)
    cleaned_df = spay_neuter(cleaned_df)
    cleaned_df = simplifying_intake_condition(cleaned_df)
    cleaned_df = simplifying_intake_type(cleaned_df)
    cleaned_df = keep_cat_dog(cleaned_df)
    print(cleaned_df["intake_type"].value_counts())



'''




Adoption  by breed 
Adoption            5810
Foster              2509
             532
Released To Wild       7
Stolen                 6
Escaped                3
           181

Dallas 
[         '',          'ADOPTION',            'FOSTER',
        '',       'LOST REPORT', 
      'FOUND REPORT',   'DEAD ON ARRIVAL',              'DIED',
           'MISSING',         'FOUND EXP',          'LOST EXP',
          'DISPOSAL',         'TREATMENT']


Long Beach 
',                  'FOSTER',
                '',         
                'ADOPTION', ,
               '',                  'RESCUE',
                       nan,               'HOMEFIRST',
           'COMMUNITY CAT',                    'DIED',
  'RETURN TO WILD HABITAT',                'DISPOSAL',
                 'MISSING',         'FOSTER TO ADOPT',
   'TRAP, NEUTER, RELEASE',               'DUPLICATE',
        'RETURN TO RESCUE']

Goal 
['Return to Owner', 'Transfer', 'Foster','Euthanasia', 'Adoption', 'Other']


def simplifying_outcome_type(df):
    
    requires col name to be ouctome_type
    
    for index in df.index:
        type = df.at[index, "outcome_type"]
        if type in ["Reclaimed","RETURNED TO OWNER","RETURN TO OWNER", 'SHELTER, NEUTER, RETURN']:
            df.at[index, "outcome_type"] = "Return to Owner"
        elif type in ["Transfer","TRANSFER","TRANSPORT",""]:
            df.at[index, "outcome_type"] = "Transfer"
        elif type in ["",""]:
            df.at[index, "outcome_type"] = "Foster"
        elif type in ["Euthanized","EUTHANIZED","EUTHANASIA"]:
            df.at[index, "outcome_type"] = "Euthanasia"
        elif type in ["",""]:
            df.at[index, "outcome_type"] = "Adoption"
        else:
            df.at[index, "outcome_type"] = "Other"

'''


main()



