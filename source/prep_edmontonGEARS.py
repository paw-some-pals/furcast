#intakes with outcome it has date of birth
import pandas as pd
import numpy as np
import matplotlib as plt
from  data_utils import categorize_color_dog, categorize_color_cat, categorize_size
import re

# -------- Load and initial renames --------
def load_data():
    '''
    Input: none
    Output: df - the raw Edmonton GEARS dataframe
    Load the Edmonton GEARS intakes with outcomes CSV
    '''
    df = pd.read_csv('datasets/intakes_by_date_with_outcomes2.csv')
    return df

def rename_columns(df):
    '''
    Input: df - dataframe with original column names
    Output: df - dataframe with renamed columns
    Rename columns to match the final dataset specifications
    '''
    df = df.rename(columns={ "SPECIESNAME": "animal_species", "BREEDNAME": "breed", "REASONNAME": "intake_type", "OUTCOMENAME": "outcome_type"})
    return df
# -------- early filtering, eda --------

def check_duplicates_and_nans(df):
    '''
    Input: df - dataframe to check for duplicates and NaN values
    Output: prints number of duplicates and NaN values in each column
    Check for duplicates and NaN values in the dataframe
    used prior to cleaning to identify any issues in the raw data that need to be addressed during cleaning
    use original df names
    '''
    num_duplicates = df.duplicated().sum()
    print(f'Number of duplicate rows: {num_duplicates}')
    
    num_nans = df.isna().sum()
    print('Number of NaN values in each column:')
    print(num_nans)

def get_unique_values(df):
    '''
    Input: df - dataframe to check for unique values
    Output: prints unique values in each column
    Check for unique values in each column to identify any inconsistencies or unexpected values that may need to be addressed during cleaning
    use original df names
    '''
    for column in df.columns:
        unique_values = df[column].unique()
        print(f'Unique values in column {column}: {unique_values}')

# ------- Remap and clean shared columns --------

def filter_intake_type(df):
    '''
    Input: df - dataframe containing intake_type
    Output: df - dataframe with only relevant intake types
    Keep only relevant intake types and map to ['Stray', 'Owner Surrender', 'Euthanasia Request', 'Other']
    Unique values in column intake_type: <StringArray>
    [  'Stray', 'Transfer from Other Organization',
      'Owner Surrender', 'Temp Care',
       'Born in Care', 'TNR - Trap/Neuter/Release',
        'SNR - Spay/Neuter/Return', 'Abuse']
    '''

    intake_type_mapping = {
        'Stray': 'Stray',
        'Transfer from Other Organization': 'Other',
        'Owner Surrender': 'Owner Surrender',
        'Temp Care': 'Other',
        'Born in Care': 'Other',
        'TNR - Trap/Neuter/Release': 'Other',
        'SNR - Spay/Neuter/Return': 'Other',
        'Abuse': 'Other'
    }

    df['intake_type'] = df['intake_type'].map(intake_type_mapping)
    df = df[df['intake_type'].isin(['Stray', 'Owner Surrender', 'Euthanasia Request', 'Other'])]
    return df


def filter_outcome_type(df):
    '''
    Input: df - dataframe containing outcome_type
    Output: df - dataframe with only relevant outcome types
    Keep only relevant outcome types and map to ['Return to Owner', 'Transfer', 'Foster','Euthanasia', 'Adoption', 'Other']
    Unique values in column outcome_type: <StringArray>
    [ 'Adoption','Transfer', 'Reclaimed',
        'Euthanasia', 'Foster', 'Released To Wild',
        'Died', 'On Shelter']
    '''

    outcome_type_mapping = {
        'Adoption': 'Adoption',
        'Transfer': 'Transfer',
        'Reclaimed': 'Return to Owner',
        'Euthanasia': 'Euthanasia',
        'Foster': 'Foster',
        'Released To Wild': 'Other',
        'Died': 'Other',
        'On Shelter': 'Other'
    }

    df['outcome_type'] = df['outcome_type'].map(outcome_type_mapping)
    df = df[df['outcome_type'].isin(['Adoption', 'Transfer', 'Foster', 'Return to Owner', 'Euthanasia', 'Other'])]
    return df

def keep_dogs_and_cats(df):
    '''
    Input: df - dataframe containing animal_species
    Output: df - dataframe containing only dogs and cats
    Keep only dog and cat rows and lowercase animal_species
    '''
    df = df[df["animal_species"].isin(["Dog", "Cat"])]
    df["animal_species"] = df["animal_species"].str.lower()
    return df

# -------- Add new features (cat and dog) --------
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

def get_seasons(month):
    if month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    elif month in [9,10,11]:
        return "Fall"
    elif month in [1,2,12]:
        return "Winter"
    
def apply_get_season(df):
    df["season"] = df["intake_month"].apply(get_seasons)
    return df

def stay_category(days):
   if pd.isna(days):
       return pd.NA
   if days <= 7:
       return "0-7 days"
   elif days <= 20:
       return "8-20 days"
   else:
       return "21+ days"

def apply_stay_category(df):
   df["stay_category"] = df["time_in_shelter"].apply(stay_category)
   return df

# TODO: add add_sex ???
# TODO: add add_neuter_status???
# TODO: intake condition???

# TODO: find edmonton data add add_population 
# TODO: add add_unemployment - merge Edmonton unemployment CSV on intake_year + intake_month

# ------- split and species specific changes --------

# TODO: add split_cat_and_dog - split into df_cat and df_dog 

# TODO: add apply_categorize_size - map breed/weight to size category using data_utils.categorize_size (dogs)
# TODO: breed map - before breed mapping, check the email from GEARS regarding affenpinscher breed comment
# TODO: add categorize_color_and_breed - map PRIMARYCOLOUR / PRIMARYBREED using data_utils helpers


# -------- Final cleanup --------
def remove_columns(df):
    '''
    Input: df - dataframe containing original and created columns
    Output: df - dataframe with unused columns removed
    Drop columns that are not needed in the final dataset
    '''
    df = df.drop(columns=[ "SHELTERCODE", "DATEOFBIRTH", "DATEBROUGHTIN", "OUTCOMEDATE"])
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


def main():
    output_path = 'datasets/temp.csv'

    df = load_data()
    df = rename_columns(df)
    df = keep_dogs_and_cats(df)
    #check_duplicates_and_nans(df)
    #get_unique_values(df)
    df = filter_intake_type(df)
    df = filter_outcome_type(df)

    df = add_age_intake(df)
    df = add_time_in_shelter(df)
    df = add_intake_date_columns(df)
    df = apply_get_season(df)
    df = apply_stay_category(df)
    
    # TODO: add_population - find Edmonton population data
    # TODO: add_unemployment - merge Edmonton unemployment CSV on intake_year + intake_month

    # TODO: add_sex and add_neuter_status, intake_condition?? knearest?

    # TODO: split_cat_and_dog - everything below runs per species
    # TODO: apply_categorize_size - dogs only
    # TODO: categorize_color_and_breed - check GEARS email re: affenpinscher before breed map

    df = remove_columns(df)
    # TODO: reorder_columns(df)

    # TODO: save_data - write final_df_gears_dogs.csv and final_df_gears_cats.csv
    df.to_csv(output_path, index=False)
    print('[DONE] Edmonton GEARS cleaning complete.')


if __name__ == '__main__':
    main()
