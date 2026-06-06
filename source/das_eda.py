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
df = df.drop(columns= ["Kennel_Status", "Activity_Sequence", "Census_Tract", "Council_District", "Reason", "Hold_Request", "Outcome_Subtype", "Outcome_Time", "Chip_Status", "Animal_Origin", "Month", "Year"])

print("------------------------------------------------------")
print(df.head())
print("------------------------------------------------------")
print(df.info())
print("------------------------------------------------------")


#convert dates to datetime
df['Intake_Date'] = pd.to_datetime(df['Intake_Date'], format='%m/%d/%Y', errors='coerce')
df['Outcome_Date'] = pd.to_datetime(df['Outcome_Date'], format='%m/%d/%Y', errors='coerce')
#calculate number of days spent in shelter
df['Shelter Time'] = df['Outcome_Date'] - df['Intake_Date']
#convert to days
df['Shelter Time'] = df['Shelter Time'].dt.days 

print(df["Shelter Time"])

print("-------------------------------------------------------")
print(f"max:", max(df["Shelter Time"]))
print(f"min:", min(df["Shelter Time"]))
print(f"mean:", df["Shelter Time"].mean())
print("-------------------------------------------------------")

#finding issue values
index_iss = df.loc[df['Shelter Time'] == -272.0, 'Index'].values[0]
index_iss2 = df.loc[df['Shelter Time'] == -55.0, 'Index'].values[0]
index_iss3 = df.loc[df['Shelter Time'] == -333.0, 'Index'].values[0]
sum_ind_iss = (df["Shelter Time"] <0).sum()
print(index_iss)
print(index_iss2)
print(index_iss3)
print("-------------------------------------------------------")

print(sum_ind_iss)
print("-------------------------------------------------------")

# get rid of these three rows
print(df["Shelter Time"][150367])
print(df["Shelter Time"][150368])
print(df["Shelter Time"][150365])

df = df.drop(index=150367)
df = df.drop(index=150368)
df = df.drop(index=150365)

# histogram of time in shelter 
hist = px.histogram(df, x="Shelter Time")
hist.show()

#pie chart of cats vs dogs vs wildlife
dog_count = (df["Animal_Type"] == 'DOG').sum()
cat_count = (df["Animal_Type"] == "CAT").sum()
wild_count = (df["Animal_Type"] == "WILDLIFE").sum()
bird_count = (df["Animal_Type"] == "BIRD").sum()
livestock_count = (df["Animal_Type"] == "LIVESTOCK").sum()

print(f"Dogs: {dog_count}, Cats: {cat_count}, Wildlife: {wild_count}")

anim_data = {"anim_counts" : [dog_count, cat_count, wild_count, bird_count, livestock_count],
              "anim_type" : ["Dog", "Cat", "Wildlife", "Bird", "Livestock"]}

animal_type = px.pie(anim_data, values='anim_counts', names='anim_type', title='Animal type breakdown')
animal_type.write_html("animal_chart.html", auto_open=True)

#make correlation chart for distribution


#scatter of shelter time vs breed
scatter_tvb = px.scatter(x= df["Shelter Time"], y=df["Animal_Breed"])
scatter_tvb.show()

#time vs type
scatter_tvt = px.scatter(x= df["Shelter Time"], y=df["Animal_Type"])
scatter_tvt.show()

#time vs intake type 
scatter_tvin = px.scatter(x= df["Shelter Time"], y=df["Intake_Type"])
scatter_tvin.show()

#time vs outtake type 
scatter_tvout = px.scatter(x=df["Shelter Time"], y=df["Outcome_Type"])
scatter_tvout.show()

#heat map of correlations between time brought in and animal type

#

#distibution of features 
