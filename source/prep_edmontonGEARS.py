#intakes with outcome it has date of birth
import pandas as pd
import numpy as np
import matplotlib as plt
import data_utils
import re

def load_data():
    '''
    Input: none
    Output: df - the raw Edmonton GEARS dataframe
    Load the Edmonton GEARS intakes with outcomes CSV
    '''
    df = pd.read_csv('datasets/intakes_by_date_with_outcomes2.csv')
    return df


def add_age_intake(df):
    '''
    Input: df - dataframe containing DATEOFBIRTH and DATEBROUGHTIN
    Output: df - dataframe with age_intake column added (in years)
    Add age_intake
    '''
    df['DATEOFBIRTH'] = pd.to_datetime(df['DATEOFBIRTH'])
    df['intake_date'] = pd.to_datetime(df['DATEBROUGHTIN'])
    df['age_intake'] = (df['intake_date'] - df['DATEOFBIRTH']).dt.days / 365.25
    return df


def add_time_in_shelter(df):
    '''
    Input: df - dataframe containing intake_date and OUTCOMEDATE
    Output: df - dataframe with time_in_shelter column added (in days)
    Add LOS; find the problematic values and print the corresponding rows; turn 0 days into 1; drop rows where time_in_shelter is NaN
    '''
    df['OUTCOMEDATE'] = pd.to_datetime(df['OUTCOMEDATE'])
    df['time_in_shelter'] = (df['OUTCOMEDATE'] - df['intake_date']).dt.days
    df['time_in_shelter'] = df['time_in_shelter'].replace(0, 1)
    nan_days = df[df['time_in_shelter'].isna()]
    #print(nan_days)
    df = df.dropna(subset=['time_in_shelter'])
    #print(df['time_in_shelter'].min())
    return df


def add_intake_date_columns(df):
    '''
    Input: df - dataframe containing intake_date
    Output: df - dataframe with intake_year, intake_month, intake_day columns added
    Change intake_date to 3 separate year, month, day columns
    '''
    df['intake_year'] = df['intake_date'].dt.year
    df['intake_month'] = df['intake_date'].dt.month
    df['intake_day'] = df['intake_date'].dt.day
    return df
 

# TODO: check duplicates?
# TODO: check only dog and cats
# TODO: add filter_intake_condition - map INTAKECONDITION to Normal / Injured / Aged / Sick / Other / Feral
# TODO: add apply_clean_intake_type - map INTAKEREASONID to Stray / Owner Surrender / Euthanasia Request / Other
# TODO: add apply_clean_outcome_type - map OUTCOMEID to Adoption / Transfer / Foster / Return to Owner / Euthanasia / Other
# TODO: add apply_get_season - derive season from intake_month 
# TODO: add apply_stay_category - bin time_in_shelter into 0-7 / 8-20 / 21+ days

# TODO: add add_sex ???
# TODO: add add_neuter_status???

# TODO: add apply_categorize_size - map breed/weight to size category using data_utils.categorize_size (dogs)
# TODO: breed map - before breed mapping, check the email from GEARS regarding affenpinscher breed comment
# TODO: add categorize_color_and_breed - map PRIMARYCOLOUR / PRIMARYBREED using data_utils helpers


# TODO: find edmonton data add add_population 
# TODO: add add_unemployment - merge Edmonton unemployment CSV on intake_year + intake_month


def remove_columns(df):
    '''
    Input: df - dataframe containing original and created columns
    Output: df - dataframe with unused columns removed
    Drop columns that are not needed in the final dataset
    '''
    df = df.drop(columns=[ "SHELTERCODE", "DATEOFBIRTH", "DATEBROUGHTIN", "OUTCOMEDATE"])
    return df

def rename_columns(df):
    '''
    Input: df - dataframe with original column names
    Output: df - dataframe with renamed columns
    Rename columns to match the final dataset specifications
    '''
    df = df.rename(columns={ "SPECIESNAME": "animal_species", "BREEDNAME": "breed", "REASONNAME": "intake_type", "OUTCOMENAME": "outcome_type"})
    return df 

def reorder_columns(df):
    df = df[[
        "age_intake",
        "sex",
        "spay_neuter",
        "intake_month",
        "intake_day",
        "intake_year",
        "animal_species",
        "colour",
        "breed",
        "intake_condition",
        "intake_type",
        "outcome_type", 
        "time_in_shelter",
    ]]
    return df

# TODO: add split_cat_and_dog - split into df_cat and df_dog 
# TODO: add save_data - write final_df_gears_dogs.csv and final_df_gears_cats.csv


def main():
    output_path = 'datasets/edmonton_gears_cleaned.csv'

    df = load_data()
    df = add_age_intake(df)
    df = add_time_in_shelter(df)
    df = add_intake_date_columns(df)


    df.to_csv(output_path, index=False)
    print('[DONE] Initial Edmonton GEARS cleaning complete.')


if __name__ == '__main__':
    main()
