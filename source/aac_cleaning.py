import pandas as pd
from data_utils import categorize_color, categorize_breed_by_species

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
    df["breed"] = df.apply(categorize_breed_by_species, axis=1)
    return df

def remove_columns(df):
    '''
    Input: df - dataframe containing original and created columns
    Output: df - dataframe with unused columns removed
    Drop columns that are not needed in the final dataset
    '''
    df = df.drop(columns=["age_upon_outcome_(years)", "age_upon_outcome_age_group", "outcome_month", "outcome_year", "outcome_monthyear", "outcome_weekday", "outcome_hour", "outcome_number", "dob_monthyear", "count", "age_upon_intake_age_group", "intake_monthyear", "intake_weekday", "intake_hour", "dob_year", "dob_month", "outcome_subtype", "age_upon_outcome", "animal_id_intake", "animal_id_outcome", "date_of_birth", "outcome_datetime", "found_location", "intake_number", "age_upon_intake", "time_in_shelter", "sex_upon_outcome", "intake_datetime", "age_upon_outcome_(days)", "age_upon_intake_(days)"])
    return df

def rename_columns(df):
    '''
    Input: df - dataframe with original column names
    Output: df - dataframe with renamed columns
    Rename columns to match the final dataset specifications
    '''
    df = df.rename(columns={"age_upon_intake_(years)": "age_intake", "neuter_status": "spay_neuter", "animal_type": "animal_species", "time_in_shelter_days": "time_in_shelter", "color": "colour"})
    return df

def reorder_columns(df):
    # reorder columns based on dataset_specifications
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
            "time_in_shelter"
        ]
    ]
    return df


def save_data(df):
    # Save cleaned dataframe to a new CSV file in the datasets folder
    df.to_csv("datasets/aac_cleaned.csv", index=False)

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
    df = remove_columns(df)
    df = rename_columns(df)
    df = reorder_columns(df)
    save_data(df)
    #print(df.columns.tolist())
    #print(df['breed'].unique())
    #print(df["breed"].value_counts())
    #print(df[["animal_species", "breed"]].head(20))

if __name__ == "__main__":
    main()