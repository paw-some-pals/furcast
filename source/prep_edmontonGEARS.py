import pandas as pd
import numpy as np
import matplotlib as plt
from  data_utils import categorize_color_dog, categorize_color_cat, categorize_size, categorize_dog_breed_by_size
import re

# -------- Load and initial renames --------
def load_data():
    '''
    Input: none
    Output: 
    df - the raw Edmonton GEARS dataframe
    df_pop - edmonton population data
    df_unemp - edmonton unemployment rate 
    df_kag_dog - kaggle breed dataset to merge with
    df_kag_cat - kaggle breed df to merge
    Load the Edmonton GEARS intakes with outcomes CSV
    '''
    df = pd.read_csv('datasets/intakes_by_date_with_outcomes2.csv')
    df_pop = pd.read_csv('datasets/edmonton_population.csv')
    df_unemp = pd.read_csv('datasets/edmonton_unemploy.csv')
    df_kag_dog = pd.read_csv('datasets/dog_breeds.csv')
    df_kag_cat = pd.read_csv('datasets/cat_breeds.csv')
    return df, df_pop, df_unemp, df_kag_dog, df_kag_cat

def rename_columns(df):
    '''
    Input: df - dataframe with original column names
    Output: df - dataframe with renamed columns
    Rename columns to match the final dataset specifications
    '''
    df = df.rename(columns={ "SPECIESNAME": "animal_species", "BREEDNAME": "breed", "REASONNAME": "intake_type", "OUTCOMENAME": "outcome_type"})
    return df

def clean_kaggle(df_kag_dog, df_kag_cat):
    """
    Drop physical measurement columns not used as model features.
    Dog key column: 'Name'. Cat key column: 'name' (renamed to 'Name' for consistency).
    """
    df_kag_dog = df_kag_dog.drop(columns=['min_life_expectancy', 'max_life_expectancy', 'max_height_male', 'max_height_female', 'max_weight_male', 'max_weight_female', 'min_height_male', 'min_height_female', 'min_weight_male', 'min_weight_female'])
    df_kag_cat = df_kag_cat.drop(columns=["length","origin"])
    df_kag_cat = df_kag_cat.rename(columns={"name": "Name"})
    return df_kag_dog, df_kag_cat


def _merge_kaggle(df, df_kag, breed_2_sentinel='not given'):
    """
    Merge df with a Kaggle breed table on breed_1 / breed_2.
    Pure breeds (breed_2 == sentinel): direct left-join on breed_1.
    Mixed breeds: average numeric Kaggle features from breed_1 and breed_2.
    """
    kaggle_feature_cols = [c for c in df_kag.columns if c != 'Name']

    pure_mask = df['breed_2'] == breed_2_sentinel

    df_pure = df[pure_mask].merge(
        df_kag, how='left', left_on='breed_1', right_on='Name'
    ).drop(columns=['Name'])

    df_mixed = df[~pure_mask].copy()

    b1_vals = df_mixed[['breed_1']].merge(
        df_kag, how='left', left_on='breed_1', right_on='Name'
    )[kaggle_feature_cols].to_numpy()

    b2_vals = df_mixed[['breed_2']].merge(
        df_kag, how='left', left_on='breed_2', right_on='Name'
    )[kaggle_feature_cols].to_numpy()

    df_mixed[kaggle_feature_cols] = (b1_vals + b2_vals) / 2

    return pd.concat([df_pure, df_mixed]).sort_index().reset_index(drop=True)


def merge_kaggle_dog(df_dog, df_kag_dog):
    return _merge_kaggle(df_dog, df_kag_dog)


def merge_kaggle_cat(df_cat, df_kag_cat):
    return _merge_kaggle(df_cat, df_kag_cat)


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

# ------- Add new features (external data) --------

def add_population(df, populations):
    '''
    Input: df - dataframe containing intake_year
    Output: df - dataframe with population column added
    Add population data for Edmonton by year
    '''
    populations = populations[["year", "population"]].rename(
        columns={"year": "intake_year"}
    )
    populations["population"] = (
        populations["population"].astype(str).str.replace(",", "", regex=False).astype(int)
    )
    return df.merge(populations, how="left", on="intake_year")

def add_unemployment(df, unemp):
    unemp["date"] = pd.to_datetime(unemp["date"])
    unemp["intake_year"] = unemp["date"].dt.year
    unemp["intake_month"] = unemp["date"].dt.month
    unemp["unemploy_rate"] = unemp["unemploy_rate"].str.replace("%", "", regex=False).astype(float)
    unemp = unemp[["intake_year", "intake_month", "unemploy_rate"]]
    return df.merge(unemp, how="left", on=["intake_year", "intake_month"])


# ------- Add new features knearest? --------
# TODO: add add_sex ???
# TODO: add add_neuter_status???
# TODO: intake condition???
# TODO: colour

# ------- split and species specific changes --------

def split_cat_and_dog(df):
    df_cat = df[df["animal_species"] == "cat"].copy()
    df_dog = df[df["animal_species"] == "dog"].copy()
    return df_cat, df_dog, df



def map_dog_breeds(df):
    '''
    Input: df - dog dataframe with a 'breed' column from Edmonton GEARS
    Output: df - dataframe with breed_1, breed_2, is_mixed columns added
    Manual mapping of GEARS breed strings to Kaggle dog_breeds.csv names.
    breed_1/breed_2 are "not given" only for truly unknown breeds (Mixed Breed, Affenpinscher artifact).
    '''
    DOG_BREED_MAP = { #LLM filled in manual map, less breeds so did not use fuzzy str match. reviewd and seems okay
        'Affenpinscher':                           ('not given','not given',1), # will need to drop
        'American Bulldog':                        ('American Bulldog',       'not given',                     0),
        'American Bulldog/Boxer':                  ('American Bulldog',       'Boxer',                         1),
        'American Staffordshire Terrier':          ('American Staffordshire Terrier', 'not given',             0),
        'Australian Cattle Dog/Blue Heeler':       ('Australian Cattle Dog',  'not given',                     0), #these are the same breed
        'Basset Hound':                            ('Basset Hound',           'not given',                     0),
        'Beagle':                                  ('Beagle',                 'not given',                     0),
        'Bernese Mountain Dog/Affenpinscher':      ('Bernese Mountain Dog',   'not given',                     1),
        'Bernese Mountain Dog/Labrador Retriever': ('Bernese Mountain Dog',   'Labrador Retriever',            1),
        'Border Collie':                           ('Border Collie',          'not given',                     0),
        'Border Collie/Beagle':                    ('Border Collie',          'Beagle',                        1),
        'Border Collie/Mixed Breed':               ('Border Collie',          'not given',                     1),
        'Boston Terrier':                          ('Boston Terrier',         'not given',                     0),
        'Boxer/Cane Corso Mastiff':                ('Boxer',                  'Cane Corso',                    1),
        'Cane Corso Mastiff':                      ('Cane Corso',             'not given',                     0),
        'Cane Corso Mastiff/Boxer':                ('Cane Corso',             'Boxer',                         1),
        'Chihuahua':                               ('Chihuahua',              'not given',                     0),
        'Cockapoo':                                ('Cocker Spaniel',         'Poodle (Miniature)',             1),
        'Dachshund':                               ('Dachshund',              'not given',                     0),
        'English Springer Spaniel/Border Collie':  ('Cocker Spaniel',         'Border Collie',                 1),
        'French Bulldog':                          ('French Bulldog',         'not given',                     0),
        'German Shepherd Dog':                     ('Anatolian Shepherd Dog', 'not given',                     0),
        'German Shepherd Dog/Mixed Breed':         ('Anatolian Shepherd Dog', 'not given',                     1),
        'Great Pyrenees':                          ('Great Pyrenees',         'not given',                     0),
        'Great Pyrenees/Mountain Dog':             ('Great Pyrenees',         'not given',                     1),
        'Havanese':                                ('Havanese',               'not given',                     0),
        'Husky':                                   ('Siberian Husky',         'not given',                     0),
        'Husky/Mixed Breed':                       ('Siberian Husky',         'not given',                     1),
        'Irish Wolfhound/Mixed Breed':             ('Greyhound',              'not given',                     1),
        'Japanese Chin':                           ('Japanese Chin',          'not given',                     0),
        'Labrador Retriever':                      ('Labrador Retriever',     'not given',                     0),
        'Labrador Retriever/Mixed Breed':          ('Labrador Retriever',     'not given',                     1),
        'Maltese':                                 ('Maltese',                'not given',                     0),
        'Mastiff':                                 ('Bullmastiff',            'not given',                     0),
        'Mastiff/Mixed Breed':                     ('Bullmastiff',            'not given',                     1),
        'Mixed Breed':                             ('not given',              'not given',                     1), #need to drop
        'Mixed Breed/Pit Bull Terrier':            ('not given',              'American Staffordshire Terrier', 1),
        'Pit Bull Terrier':                        ('American Staffordshire Terrier', 'not given',             0),
        'Poodle':                                  ('Poodle (Miniature)',     'not given',                     0),
        'Poodle/Pomeranian':                       ('Poodle (Miniature)',     'Pomeranian',                    1),
        'Rottweiler':                              ('Rottweiler',             'not given',                     0),
        'Shepherd':                                ('Anatolian Shepherd Dog', 'not given',                     0),
        'Shepherd/Mixed Breed':                    ('Anatolian Shepherd Dog', 'not given',                     1),
        'Shih Tzu':                                ('Shih Tzu',               'not given',                     0),
        'Spaniel':                                 ('Cocker Spaniel',         'not given',                     0),
        'Xoloitzcuintle/Mexican Hairless':         ('Xoloitzcuintli',        'not given',                     0),
        'Yorkshire Terrier Yorkie':                ('Yorkshire Terrier',      'not given',                     0),
    }
    df['breed'] = df['breed'].str.strip()
    mapped = df['breed'].map(DOG_BREED_MAP)
    df['breed_1'] = mapped.apply(lambda x: x[0] if isinstance(x, tuple) else 'not given')
    df['breed_2'] = mapped.apply(lambda x: x[1] if isinstance(x, tuple) else 'not given')
    df['is_mixed'] = mapped.apply(lambda x: x[2] if isinstance(x, tuple) else 0)
    df = df[df['breed_1'] != 'not given']
    return df


def map_cat_breeds(df):
    '''
    Input: df - cat dataframe with a 'breed' column from Edmonton GEARS
    Output: df - dataframe with breed_1, breed_2, is_mixed columns added
    Manual mapping of GEARS breed strings to Kaggle cat_breeds.csv names.
    Notes:
      - "Domestic Long/Medium/Short Hair" are coat descriptors, not breeds -> not given.
      - Kaggle uses "Bengal Cats", "Ragdoll Cats", "Siamese Cat" (with suffix).
    '''
    CAT_BREED_MAP = {
        'Bengal':               ('Bengal Cats',   'not given', 0),
        'Domestic Long Hair':   ('American Longhair',     'not given', 0),
        'Domestic Medium Hair': ('American Longhair',     'not given', 0),
        'Domestic Short Hair':  ('American Shorthair',     'not given', 0),
        'Maine Coon':           ('Maine Coon',    'not given', 0),
        'Manx':                 ('Manx',          'not given', 0),
        'Ragdoll':              ('Ragdoll Cats',  'not given', 0),
        'Scottish Fold':        ('Scottish Fold', 'not given', 0),
        'Siamese':              ('Siamese Cat',   'not given', 0),
    }
    df['breed'] = df['breed'].str.strip()
    mapped = df['breed'].map(CAT_BREED_MAP)
    df['breed_1'] = mapped.apply(lambda x: x[0] if isinstance(x, tuple) else 'not given')
    df['breed_2'] = mapped.apply(lambda x: x[1] if isinstance(x, tuple) else 'not given')
    df['is_mixed'] = mapped.apply(lambda x: x[2] if isinstance(x, tuple) else 0)
    df = df[df['breed_1'] != 'not given']
    return df


def apply_categorize_size(df_dog):
    '''
    Input: df_dog - dog-only dataframe with breed_1 column (mapped Kaggle breed name)
    Output: df_dog - dataframe with animal_size column added
    Apply size categorization using breed_1 after map_dog_breeds has run.
    '''
    df_dog['animal_size'] = df_dog['breed_1'].apply(categorize_dog_breed_by_size)
    return df_dog


# TODO: cat: black/white indicator


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
        #"sex",
        #"spay_neuter",
        "intake_month",
        "intake_day",
        "intake_year",
        "animal_species",
        #"colour",
        "breed",
        #"intake_condition",
        "intake_type",
        "season",
        "outcome_type", 
        "time_in_shelter",
        "stay_category",
    ]]
    return df


def main():
    output_path_full = 'datasets/final_df_gears.csv'
    output_dog= 'final_df_gears_dogs.csv'
    output_cat ='final_df_gears_cats.csv'

    df, df_pop, df_unemp, df_kag_dog, df_kag_cat = load_data()
    df_kag_dog, df_kag_cat = clean_kaggle(df_kag_dog, df_kag_cat)

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
    df = add_population(df, df_pop)
    df = add_unemployment(df, df_unemp)
    # TODO: add_sex and add_neuter_status, intake_condition?? knearest?

    df = remove_columns(df)
    df = reorder_columns(df)

    df_cat, df_dog, df = split_cat_and_dog(df)
    # add colour??? knearest?

    df_dog = map_dog_breeds(df_dog)
    df_cat = map_cat_breeds(df_cat)
    #print(df_dog["breed"].unique())
    #print(df_cat["breed"].unique())

    df_dog = apply_categorize_size(df_dog)
    df_dog = merge_kaggle_dog(df_dog, df_kag_dog)
    df_cat = merge_kaggle_cat(df_cat, df_kag_cat)
    # TODO: black/white indicator (NEED TO ADD COLOUR FIRST)

    df.to_csv(output_path_full, index=False)
    df_dog.to_csv(output_dog, index=False)
    df_cat.to_csv(output_cat, index=False)
    print('[DONE] Edmonton GEARS cleaning complete. See: final_df_gears.csv, final_df_gears_dogs.csv, final_df_gears_cats.csv')


if __name__ == '__main__':
    main()
