#Dallas Animal Shelter cleaning

#columns
#Animal_Id	Animal_Type	Animal_Breed	Kennel_Status	Activity_Sequence	Census_Tract	Council_District	Intake_Type	Intake_Subtype	Reason	Intake_Date	Intake_Time	Intake_Condition	Hold_Request	Outcome_Type	Outcome_Subtype	Outcome_Date	Outcome_Time	Outcome_Condition	Chip_Status	Animal_Origin	Month	Year

#csv columns to keep
#ID, Animal Type, Breed, Intake Type, Intake Subtype, Reason, Intake Date, Intake condition, outcome condition, outcome type, outcome date, chip status

import pandas as pd 
import data_utils

df = pd.read_csv('datasets/dallas/shelterdata2017-2021.csv')

df = df.drop(columns= ["Kennel_Status", "Activity_Sequence", "Census_Tract", "Council_District", "Reason", "Hold_Request", "Outcome_Subtype", "Intake_Time", "Outcome_Time", "Animal_Origin", "Month", "Year"])

#convert dates to datetime
df['Intake_Date'] = pd.to_datetime(df['Intake_Date'], format='%m/%d/%Y', errors='coerce')
df['Outcome_Date'] = pd.to_datetime(df['Outcome_Date'], format='%m/%d/%Y', errors='coerce')
#calculate number of days spent in shelter
df['Shelter Time'] = df['Outcome_Date'] - df['Intake_Date']
#convert to days
df['Shelter Time'] = df['Shelter Time'].dt.days 
#remove negative times 
df = df[df["Shelter Time"] >= 0]

print("-------------------------------------------------------")
print(f"max:", max(df["Shelter Time"]))
print(f"min:", min(df["Shelter Time"]))
print(f"mean:", df["Shelter Time"].mean())
print("-------------------------------------------------------")

# convert intake date to day month year 

#keep only dog and cat columns 
df.rename(columns={'Animal_Type': 'animal_species'}, inplace=True) # make consistent with other datasets
df['animal_species'] = df['animal_species'].str.lower()
df = data_utils.simplify_animal_species(df)

#drop treatment columns and euthenasia columns 
df = df[df["Intake_Type"] != "TREATMENT"]
df = df[df['Intake_Subtype'] != "TREATMENT"]
df = df[df["Intake_Subtype"] != "EUTHANASIA REQUESTED"]
df = df[df["Intake_Subtype"] != "'- DEAD ON ARRIVAL"]


counts = df['Animal_Id'].value_counts()

# Filter out values that only appear once to isolate the duplicates
repeated_counts = counts[counts > 1]
print(repeated_counts)
print("-------------------------------------------------------")

print(df[df['Animal_Id'] == 'A1084959'])

#def split_date(df):


