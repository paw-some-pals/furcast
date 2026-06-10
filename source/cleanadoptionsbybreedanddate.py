import pandas as pd
import data_utils
import seaborn as sns
df = pd.read_csv('datasets/animal-data-1.csv')
print(df.head())
print (df.info())

ogdf=df.copy()

#code from google ai and claude
#function converts age string to floating point value
import re

def age_to_float(age_str):
    if pd.isna(age_str):
        return None

    # Ensure we are working with a string
    age_str = str(age_str)

    # Find all integers in the string
    numbers = re.findall(r'\d+', age_str)
    
    # Extract years and months (defaulting to 0 if not present)
    years = int(numbers[0]) if len(numbers) > 0 else 0
    months = int(numbers[1]) if len(numbers) > 1 else 0
    
    # Calculate and return float
    return float(years + (months / 12))

# Test
print(age_to_float("5 years 3 months"))  # Output: 5.25
print(age_to_float("7m"))                # Output: 0.5833333333333334

# Convert full column and add results to the DataFrame
df['animal_age_float'] = df['animalage'].apply(age_to_float)
print(df[['animalage', 'animal_age_float']].head())

#drop columns that arent necessary
df=df.drop(columns=['index','animalage','istransfer','sheltercode','identichipnumber','animalname','location','istrial','returndate','returnedreason','deceaseddate','deceasedreason','diedoffshelter','isdoa'])
print(list(df.columns))

df=df.rename(columns={'speciesname': 'animal_species'})
df['animal_species'] = df['animal_species'].str.lower()
df=data_utils.simplify_animal_species(df)
print('------------------------------')

#jenna use this line of code to find euthanasia outcomes but after you drop all your unneeded rows
euthanasia_count = df[df['puttosleep'] == 1]
print(euthanasia_count)

print("----------------------------------------------------------------------------------------")
#change movement type to 'euthanasia' if puttosleep is 1, but only for unique values in the id column
id_counts = df['id'].value_counts()
df.loc[(df['puttosleep'] == 1) & (df['id'].map(id_counts) == 1), 'movementtype'] = 'Euthanasia'
print(df[df['puttosleep'] == 1][['id', 'puttosleep', 'movementtype']].head(15))

#jenna dont use this line, as it only keep a unique ids first intake date
df = df.drop_duplicates(subset=['id'], keep='first')
#print(df)



print(df)
#TODO add time stayed column in days
df['intakedate'] = pd.to_datetime(df['intakedate'])
df['movementdate'] = pd.to_datetime(df['movementdate'])

df['days_in_shelter'] = (df['movementdate'] - df['intakedate']).dt.days
#print(df[['id', 'intakedate', 'movementdate', 'days_in_shelter']].head())

shelter_minimum = df['days_in_shelter'].min()
shelter_maximum = df['days_in_shelter'].max()
#print(f"Minimum days in shelter: {shelter_minimum}")
#print(f"Maximum days in shelter: {shelter_maximum}")

#find negative values and their rows
#find the problematic negative minimum value and print the corresponding rows
negative_days = df[df['days_in_shelter'] < 0]
#print(negative_days[['id', 'intakedate', 'movementdate', 'days_in_shelter']])

messed_up_intake_values = df[df['days_in_shelter'] < 0]
#print(messed_up_intake_values[['id', 'intakedate', 'movementdate', 'days_in_shelter']])
for row in messed_up_intake_values.itertuples():
    df.at[row.Index, 'intakedate'] = row.movementdate
    df.at[row.Index, 'movementdate'] = row.intakedate
df['days_in_shelter'] = (df['movementdate'] - df['intakedate']).dt.days
#print(df[['id', 'intakedate', 'movementdate', 'days_in_shelter']].head())

#TODO convert the intakedate and movement date to 6 separate intakeyear intake month intake day and same for movement
df["intakeyear"] = df["intakedate"].dt.year
df["intakemonth"] = df["intakedate"].dt.month
df["intakedown"] = df["intakedate"].dt.day
df["movementyear"] = df["movementdate"].dt.year
df["movementmonth"] = df["movementdate"].dt.month
df["movementday"] = df["movementdate"].dt.day



df=df.drop(columns=['intakedate', 'movementdate'])
print('----')
#print(df.columns)

df=df.rename(columns={'animal_age_float': 'age_intake','movementtype':'outcome_type', 'days_in_shelter': 'time_in_shelter', 'sexname':'sex', 'intakeyear':'intake_year', 'intakemonth':'intake_month', 'intakedown':'intake_day', 'movementyear':'movement_year', 'movementmonth':'movement_month', 'movementday':'movement_day', 'basecolour':'colour', 'breedname':'breed'})
df=df._rename(columns={'intakereason':'intake_type'})
#print(df.columns)
print('--------------------')
pd.set_option('display.max_columns', None)
print(df)

#intake_type: ['Stray', 'Owner Surrender', 'Euthanasia Request', 'Other']
print(df['intake_type'].value_counts())
#'Stray' just maps naturally, also map police assist to stray, transfer from other shelter, born in shelter to stray
#map 'Incompatible with owner lifestyle', 'litter relinquishment', 'moving', 'unsuitable accomodation', 'unable to afford', 'abandoned, 'landlord issues', 'Owner deceased', 'sick/injured', 'allergie' , 'incompatible with other pets', 'biting', 'rabies monitoring',

import os
import plotting
os.makedirs('figures', exist_ok=True)
mi_df = plotting.mutual_info_regression_matrix(ogdf.dropna())
plotting.plot_mutual_info_heatmap(mi_df, save_path='figures/ogdfABD_MI_heatmap.png')











