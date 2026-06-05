#Dallas Animal Shelter EDA

#columns
#Animal_Id	Animal_Type	Animal_Breed	Kennel_Status	Activity_Sequence	Census_Tract	Council_District	Intake_Type	Intake_Subtype	Reason	Intake_Date	Intake_Time	Intake_Condition	Hold_Request	Outcome_Type	Outcome_Subtype	Outcome_Date	Outcome_Time	Outcome_Condition	Chip_Status	Animal_Origin	Month	Year


#csv columns to keep
#ID, Animal Type, Breed, Intake Type, Intake Subtype, Reason, Intake Date, Intake condition, outcome type, outcome date 



import pandas as pd 
import plotly.express as px

from datetime import datetime

df = pd.read_csv('datasets/dallas/shelterdata2017-2021.csv')

#get rid of uneeded data 
df = df.drop(columns= ["Kennel_Status", "Activity_Sequence", "Census_Tract", "Council_District", "Reason", "Intake_Time", "Hold_Request", "Outcome_Subtype", "Outcome_Time", "Chip_Status", "Animal_Origin", "Month", "Year"])

print("------------------------------------------------------")
print(df.head())
print("------------------------------------------------------")
print(df.info())
print("------------------------------------------------------")


#calculate time in shelter and make new column

df['Intake_Date'] = pd.to_datetime(df['Intake_Date'], format='%m/%d/%Y', errors='coerce')
df['Outcome_Date'] = pd.to_datetime(df['Outcome_Date'], format='%m/%d/%Y', errors='coerce')
df['Shelter Time'] = df['Outcome_Date'] - df['Intake_Date']
df['Shelter Time'] = df['Shelter Time'].dt.days

print(df["Shelter Time"])

hist = px.histogram(df, x="Shelter Time")
hist.show()