
#imports
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
from data_utils import simplify_animal_species, simplifying_intake_type, categorize_color, split_date
import plotting


#only keeping cat and dog using function from utils 
def keep_cat_dog(df):
    df.rename(columns={'Animal Type': 'animal_species'}, inplace=True) # make consistent with other datasets
    df['animal_species'] = df['animal_species'].str.lower()
    new_df = simplify_animal_species(df)  
    return new_df   


def read_file(path):
    lb_df = pd.read_csv("datasets/animal-shelter-intakes-and-outcomes.csv")
    return lb_df

def remove_columns(df):
    # removing the columns we will not be using
    df = df.drop([
        "Kennel ID","latitude", "longitude", "Animal Name", "Secondary Color",
        "Intake Subtype", "Reason for Intake", "Crossing", "Jurisdiction",
        "Outcome Subtype", "outcome_is_dead", "was_outcome_alive", "geopoint",
        "is_current_month", "outcome_is_current","DOB", "Outcome Date",
        "outcome_is_other", "outcome_is_alive", "intake_is_dead"
    ], axis=1)

    return df 

def calc_age(df):
    # age added in years float
    age_in_days = pd.to_datetime(df["intake_date"]) - pd.to_datetime(df["DOB"])
    df["age_intake"] = (age_in_days / pd.to_timedelta(365.4525, "D")).round(2)

def remove_duplicates(df):
    total_duplicates = df.duplicated().sum() #THIS GIVES 30 
    df.drop_duplicates(inplace=True)  #dropping rows that are fully duplicated like all cols same 

    

def create_long_short_datasets(df):
   df_sorted = df.sort_values(by=['animal_id', 'intake_date'])
   df_first = df_sorted.drop_duplicates(subset=['animal_id'], keep='first')
   df_last = df_sorted.drop_duplicates(subset=['animal_id'], keep='last')
   df_last.drop(columns=["animal_id", "intake_date"],inplace=True)
   df_first.drop(columns=["animal_id", "intake_date"],inplace=True)

   df_first.to_csv("datasets/long_beach_short_term.csv", index=False)
   df_last.to_csv("datasets/long_beach_long_term.csv", index=False)

   return df_last, df_first




def drop_missing_values(df):
    df.dropna(subset=["Outcome Date", "DOB", "Outcome Type"], inplace=True)

def spay_neuter(df):
    for index in df.index:
        sex = df.at[index, "sex"]
        if sex == "Male":
            df.at[index, "sex"] = "Male"
            df.at[index, "spay_neuter"] = "No"
        elif sex == "Female":
            df.at[index, "sex"] = "Female"
            df.at[index, "spay_neuter"] = "No"
        elif sex == "Neutered":
            df.at[index, "sex"] = "Male"
            df.at[index, "spay_neuter"] = "Yes"
        elif sex == "Spayed":
            df.at[index, "sex"] = "Female"
            df.at[index, "spay_neuter"] = "Yes"
        else:
            df.at[index, "sex"] = "Unknown"
            df.at[index, "spay_neuter"] = "Unknown"

    return df


# # all ill mapped to sick
# # underage/weight mapped to sick
# # everything else is other 
def simplifying_intake_condition(df):
    for index in df.index:
        condition = df.at[index, "intake_condition"]
        if condition == "NORMAL":
            df.at[index, "intake_condition"] = "Normal"
        elif "INJURED" in condition:
            df.at[index, "intake_condition"] = "Injured"
        elif "ILL" in condition or condition == "UNDER AGE/WEIGHT":
            df.at[index, "intake_condition"] = "Sick"
        elif condition == "FERAL":
            df.at[index, "intake_condition"] = "Feral"
        elif condition == "AGED":
            df.at[index, "intake_condition"] = "Aged"
        else:
            df.at[index, "intake_condition"] = "Other"
    return df 

def changing_col_names(df):
    df.rename(columns={'Animal Type': 'animal_species'}, inplace=True)
    df.rename(columns={'Animal ID': 'animal_id'}, inplace=True)
    df.rename(columns={'Primary Color': 'colour'}, inplace=True)
    df.rename(columns={'Sex': 'sex'}, inplace=True)
    df.rename(columns={'Intake Date': 'intake_date'}, inplace=True)
    df.rename(columns={'Intake Condition': 'intake_condition'}, inplace=True)
    df.rename(columns={'Intake Type': 'intake_type'}, inplace=True)
    df.rename(columns={'Outcome Type': 'outcome_type'}, inplace=True)
    df.rename(columns={'intake_duration': 'time_in_shelter'}, inplace=True)
    return df


def simplifying_outcome_type(df):
    '''
    requires col name to be ouctome_type
    Goal Outcome Type= ['Return to Owner', 'Transfer', 'Foster','Euthanasia', 'Adoption', 'Other']
    Input: dataframe
    Output: updated dataframe with the outcome_type col simplified 
    '''
    
    for index in df.index:
        type = df.at[index, "outcome_type"]
        if type in ["Reclaimed","RETURNED TO OWNER","RETURN TO OWNER", 'SHELTER, NEUTER, RETURN']:
            df.at[index, "outcome_type"] = "Return to Owner"
        elif type in ["Transfer","TRANSFER","TRANSPORT","RESCUE"]:
            df.at[index, "outcome_type"] = "Transfer"
        elif type in ["Foster","FOSTER","FOSTER TO ADOPT"]:
            df.at[index, "outcome_type"] = "Foster"
        elif type in ["Euthanized","EUTHANIZED","EUTHANASIA"]:
            df.at[index, "outcome_type"] = "Euthanasia"
        elif type in ["Adoption","ADOPTION"]:
            df.at[index, "outcome_type"] = "Adoption"
        else:
            df.at[index, "outcome_type"] = "Other"
    return df
  
def change_zeros(df):
    df["time_in_shelter"] = df["time_in_shelter"].replace(0, 1)

def change_color(df):
    df["colour"] = df["colour"].apply(categorize_color)

def intake_date_split(df):
    #changing str to datetime
    df["intake_date"] = pd.to_datetime(df["intake_date"])
    split_date(df,"intake_date")
    df.rename(columns={'year': 'intake_year'}, inplace=True)
    df.rename(columns={'month': 'intake_month'}, inplace=True)
    df.rename(columns={'day': 'intake_day'}, inplace=True)


    return df 

   

def main():

    my_df = read_file("datasets/animal-shelter-intakes-and-outcomes.csv")
    drop_missing_values(my_df)
    cleaned_df = keep_cat_dog(my_df)
    cleaned_df = changing_col_names(cleaned_df)

    #adding cols
    calc_age(cleaned_df)
    cleaned_df = spay_neuter(cleaned_df)

    #removing cols
    cleaned_df = remove_columns(cleaned_df)

    remove_duplicates(cleaned_df)
    change_zeros(cleaned_df)

    cleaned_df = simplifying_intake_condition(cleaned_df)
    cleaned_df = simplifying_intake_type(cleaned_df)
    cleaned_df = simplifying_outcome_type(cleaned_df)
    change_color(cleaned_df)
    cleaned_df = intake_date_split(cleaned_df)

    lb_long_cleaned, lb_short_cleaned = create_long_short_datasets(cleaned_df)

    plotting.mutual_info_regression_matrix(lb_long_cleaned,filename="figures/LB_heatmap.png")

    
main()



