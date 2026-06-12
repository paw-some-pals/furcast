import pandas as pd
from data_utils import categorize_color, categorize_breed_by_species, categorize_size
from plotting import mutual_info_regression_matrix, plot_mutual_info_heatmap

def load_data():
    df = pd.read_csv("datasets/aac_intakes_outcomes.csv")
    return df

def remove_duplicates(df):
    '''
    Input: df - the original dataframe
    Output: df - dataframe with duplicate rows removed
    '''
    #Drop duplicates
    df = df.drop_duplicates()
    return df

def change_foster_adoptions(df):
    '''
    Input: df - dataframe containing outcome_type and outcome_subtype
    Output: df - dataframe where foster adoptions are labeled as Foster
    Change foster adoptions to foster and drop outcome_subtype
    '''
    foster_adoptions = (df["outcome_type"] == "Adoption") & (df["outcome_subtype"] == "Foster")
    df.loc[foster_adoptions, "outcome_type"] = "Foster"
    return df

def convert_time_in_shelter(df):
    '''
    Input: df - dataframe containing time_in_shelter
    Output: df - dataframe with time_in_shelter_days as a numeric column
    Convert time_in_shelter into total days
    '''
    # Convert time_in_shelter to timedelta format
    df["time_in_shelter"] = pd.to_timedelta(df["time_in_shelter"])
    df["time_in_shelter_days"] = df["time_in_shelter"].dt.total_seconds() / (60 * 60 * 24) # convert the time into days
    return df

def convert_age_to_float(df):
    '''
    Input: df - dataframe containing age_upon_intake_(years)
    Output: df - dataframe where age_upon_intake_(years) is a float
    Convert intake age to a numeric float value
    '''
    # change age_upon_intake to a float
    df['age_upon_intake_(years)'] = df['age_upon_intake_(years)'].astype(float)
    return df

def neuter_status(sex):
    '''
    Input: sex - value from sex_upon_intake
    Output: string - Yes, No, or Unknown
    Categorize whether the animal is spayed/neutered
    '''
    if "Neutered" in str(sex) or "Spayed" in str(sex):
        return "Yes"
    elif "Intact" in str(sex):
        return "No"
    else:
        return "Unknown"

def add_neuter_status(df):
    '''
    Input: df - dataframe containing sex_upon_intake
    Output: df - dataframe with a new neuter_status column
    Add simplified spay/neuter status to the dataframe
    '''
    df["neuter_status"] = df["sex_upon_intake"].apply(neuter_status)
    return df

def sex_status(sex):
    '''
    Input: sex - value from sex_upon_intake
    Output: string - Male, Female, or Unknown
    Categorize the animal's sex
    '''    
    if "Male" in str(sex):
        return "Male"
    elif "Female" in str(sex):
        return "Female"
    else:
        return "Unknown"
    
def add_sex(df):
    # Create a new column for sex status
    df["sex"] = df["sex_upon_intake"].apply(sex_status)
    return df

def filter_intake_condition(df):
    '''
    Input: df - dataframe containing intake_condition
    Output: df - dataframe filtered to selected intake conditions
    Remove intake conditions that are not part of the final categories
    '''
    df = df[df["intake_condition"].isin(["Normal", "Injured", "Aged", "Sick", "Other", "Feral"])]
    return df

def clean_intake_type(intake_type):
    '''
    Input: intake_type - value from intake_type column
    Output: string - cleaned intake type category
    Group intake types into selected categories or Other
    '''
    if intake_type in ["Stray", "Owner Surrender", "Euthanasia Request"]:
        return intake_type
    else:
        return "Other"
    
def apply_clean_intake_type(df):
    '''
    Input: df - dataframe containing intake_type
    Output: df - dataframe with cleaned intake_type values
    Apply intake type cleaning to the dataframe
    '''
    df["intake_type"] = df["intake_type"].apply(clean_intake_type)
    return df

def clean_outcome_type(outcome_type):
    '''
    Input: outcome_type - value from outcome_type column
    Output: string - cleaned outcome type category
    Group outcome types into selected categories or Other
    '''
    if outcome_type in ["Return to Owner", "Transfer", "Foster", "Euthanasia", "Adoption"]:
        return outcome_type
    else:
        return "Other"
    
def apply_clean_outcome_type(df):
    #Apply outcome type cleaning to the dataframe
    df["outcome_type"] = df["outcome_type"].apply(clean_outcome_type)
    return df

def keep_dogs_and_cats(df):
    '''
    Input: df - dataframe containing animal_type
    Output: df - dataframe containing only dogs and cats
    Keep only dog and cat rows and lowercase animal_type
    '''
    df = df[df["animal_type"].isin(["Dog", "Cat"])]
    df["animal_type"] = df["animal_type"].str.lower()
    return df

def add_intake_day(df):
    '''
    Input: df - dataframe containing intake_datetime
    Output: df - dataframe with intake_day added
    Convert intake_datetime and extract the intake day
    '''
    df["intake_datetime"] = pd.to_datetime(df["intake_datetime"])
    df['intake_day'] = df['intake_datetime'].dt.day
    return df

def categorize_color_and_breed(df):
    '''
    Input: df - dataframe containing color, breed, and animal_type
    Output: df - dataframe with categorized color and breed values
    Apply shared color and breed categorization functions
    '''
    
    df["color"] = df["color"].apply(categorize_color)
    #df["breed"] = df.apply(categorize_breed_by_species, axis=1)

    return df
    
def apply_categorize_size(df):
    df["animal_size"] = df.apply(categorize_size, axis=1)
    return df

def remove_columns(df):
    '''
    Input: df - dataframe containing original and created columns
    Output: df - dataframe with unused columns removed
    Drop columns that are not needed in the final dataset
    '''
    df = df.drop(columns=["age_upon_outcome_(years)", "age_upon_outcome_age_group", "outcome_month", "outcome_year", "outcome_monthyear", "outcome_weekday", "outcome_hour", "outcome_number", "dob_monthyear", "count", "age_upon_intake_age_group", "intake_monthyear", "intake_weekday", "intake_hour", "dob_year", "dob_month", "outcome_subtype", "age_upon_outcome", "animal_id_intake", "animal_id_outcome", "date_of_birth", "outcome_datetime", "found_location", "age_upon_intake", "time_in_shelter", "sex_upon_outcome", "intake_datetime", "age_upon_outcome_(days)", "age_upon_intake_(days)", "intake_number"])
    return df

def rename_columns(df):
    '''
    Input: df - dataframe with original column names
    Output: df - dataframe with renamed columns
    Rename columns to match the final dataset specifications
    '''
    df = df.rename(columns={"age_upon_intake_(years)": "age_intake", "neuter_status": "spay_neuter", "animal_type": "animal_species", "time_in_shelter_days": "time_in_shelter", "color": "colour",})
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
        "animal_size",
        "colour",
        "breed",
        "intake_condition",
        "intake_type",
        "outcome_type", 
        "time_in_shelter",
    ]]
    return df

def split_cat_and_dog(df):
    df_cat = df[df["animal_species"] == "cat"].copy()
    df_dog = df[df["animal_species"] == "dog"].copy()
    return df_cat, df_dog, df


def save_data(df_cat, df_dog, df):
    # Save cleaned dataframe to a new CSV file in the datasets folder
    #df.to_csv("datasets/aac_cleaned.csv", index=False)
    df_cat.to_csv("datasets/aac_cat_cleaned.csv", index = False)
    df_dog.to_csv("datasets/aac_dog_cleaned.csv", index = False)

def clean_kaggle():
    df = pd.read_csv("datasets/dog_breeds.csv")
    df = df.drop(columns=['min_life_expectancy', 'max_life_expectancy', 'max_height_male', 'max_height_female', 'max_weight_male', 'max_weight_female', 'min_height_male', 'min_height_female', 'min_weight_male', 'min_weight_female'])
    return df

def fill_values(df_kaggle):
    df = pd.read_csv("datasets/acc_dog_breed_split.csv")
    
    kaggle_feature_cols = [c for c in df_kaggle.columns if c != 'Name']

    pure_mask = df['breed_2'] == 'None'

    # Pure breeds: merge directly on breed_1
    df_pure = df[pure_mask].merge(
        df_kaggle, how='left', left_on='breed_1', right_on='Name'
    ).drop(columns=['Name'])

    # Mixed breeds: average kaggle features for breed_1 and breed_2
    df_mixed = df[~pure_mask].copy()

    b1_vals = df_mixed[['breed_1']].merge(
        df_kaggle, how='left', left_on='breed_1', right_on='Name'
    )[kaggle_feature_cols].to_numpy()

    b2_vals = df_mixed[['breed_2']].merge(
        df_kaggle, how='left', left_on='breed_2', right_on='Name'
    )[kaggle_feature_cols].to_numpy()

    df_mixed[kaggle_feature_cols] = (b1_vals + b2_vals) / 2

    return pd.concat([df_pure, df_mixed]).sort_index().reset_index(drop=True)



def heatmap(df):

    df_plot = df.copy()

    # Select interpretable features  (drop IDs, raw datetimes, string age columns, features with many unique values)
    heatmap_cols = [
        "age_intake",
        "sex",
        "spay_neuter",
        "intake_month",
        "intake_day",
        "intake_year",
        "animal_species",
        "animal_size",
        "colour",
        "breed",
        "intake_condition",
        "intake_type",
        "outcome_type", 
        "time_in_shelter",
        ]
    df_heatmap = df[heatmap_cols].dropna()

    mi_df = mutual_info_regression_matrix(df_heatmap, filename="figures/aac_cleaned_heatmap.png")

    return mi_df

def main():
    df = load_data()

    df = remove_duplicates(df)
    df = change_foster_adoptions(df)
    df = convert_time_in_shelter(df)
    df = convert_age_to_float(df)
    df = add_neuter_status(df)
    df = add_sex(df)
    df = filter_intake_condition(df)
    df = apply_clean_intake_type(df)
    df = apply_clean_outcome_type(df)
    df = keep_dogs_and_cats(df)
    df = add_intake_day(df)
    df = categorize_color_and_breed(df)
    df = apply_categorize_size(df)
    df = remove_columns(df)
    df = rename_columns(df)
    df = reorder_columns(df)
    df_cat, df_dog, df = split_cat_and_dog(df)
    save_data(df_cat, df_dog, df)
    df_kaggle = clean_kaggle()
    final_df_aac_dogs = fill_values(df_kaggle)
    final_df_aac_dogs.to_csv("datasets/final_df_aac_dogs.csv", index = False)


    heatmap(df)
    #print(df.columns.tolist())
    #print(df['breed'].unique())
    #print(df["animal_size"].value_counts())
    #print(df[["animal_species", "breed"]].head(20))

if __name__ == "__main__":
    main()