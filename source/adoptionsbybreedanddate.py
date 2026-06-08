import pandas as pd
df = pd.read_csv('datasets/animal-data-1.csv')
print(df.head())
print (df.info())

animal_age=df['animalage']
print(animal_age.head())

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

#drop the following columns
df=df.drop(columns=['index','animalage','istransfer','sheltercode','identichipnumber','animalname','location','istrial','returndate','returnedreason','deceaseddate','deceasedreason','diedoffshelter','isdoa'])
print(list(df.columns))

#need to add time stayed column in days
# only for rows with unique values in id column, and take the row with the most recent date if there are multiple rows with the same id
#also because the same intake date is used for an animal with multiple rows
#first days spent in shelter will be calculated by taking the difference between the most recent date and the intake date, then converting to days

df['intakedate'] = pd.to_datetime(df['intakedate'])
df['movementdate'] = pd.to_datetime(df['movementdate'])

# Per animal: find the most recent movementdate and the shared intakedate
time_in_shelter = (
    df.groupby('id')
    .agg(
        most_recent_movementdate=('movementdate', 'max'),
        intakedate=('intakedate', 'first')
    )
    .reset_index()
)

time_in_shelter['days_in_shelter'] = (
    time_in_shelter['most_recent_movementdate'] - time_in_shelter['intakedate']
).dt.days

df = df.merge(time_in_shelter[['id', 'days_in_shelter']], on='id', how='left')
print(df[['id', 'intakedate', 'movementdate', 'days_in_shelter']].head(10))

#use days in shelter and age upon intake to calculate age at outcome
df['age_at_outcome'] = df['animal_age_float'] + (df['days_in_shelter'] / 365)
print(df[['animal_age_float', 'days_in_shelter', 'age_at_outcome']].head(10))

#calculate average days in shelter but only for unique values in id
average_days_in_shelter = df.groupby('id')['days_in_shelter'].first().mean()
print(f'Average days in shelter: {average_days_in_shelter:.2f}')

#min and max days stayed
min_days = df.groupby('id')['days_in_shelter'].first().min()
max_days = df.groupby('id')['days_in_shelter'].first().max()
print(f'minimum days in shelter: {min_days}')
print(f'maximum days in shelter: {max_days}')

#find the problematic negative minimum value and print the corresponding rows
negative_days = df[df['days_in_shelter'] < 0]
print(negative_days[['id', 'intakedate', 'movementdate', 'days_in_shelter']])

#find all negative values including -1 and print the corresponding rows
other_negative_days = df[df['days_in_shelter'] < 0]
print(other_negative_days[['id', 'intakedate', 'movementdate', 'days_in_shelter']])

#other entries are due to input errors, will swap the movementdate and intakedate for those entries and append our dataset


for row in other_negative_days.itertuples():
    df.at[row.Index, 'intakedate'] = row.movementdate
    df.at[row.Index, 'movementdate'] = row.intakedate

#recalculate days in shelter after fixing negative days
df['days_in_shelter'] = (df['movementdate'] - df['intakedate']).dt.days

#catch any new negatives exposed by the per-row recalculation and swap their dates too
new_negatives = df[df['days_in_shelter'] < 0]
for row in new_negatives.itertuples():
    df.at[row.Index, 'intakedate'] = row.movementdate
    df.at[row.Index, 'movementdate'] = row.intakedate
df['days_in_shelter'] = (df['movementdate'] - df['intakedate']).dt.days
print(df[['id', 'intakedate', 'movementdate', 'days_in_shelter']].head(10))

#recalcualte age at outcome after fixing negative days
df['age_at_outcome'] = df['animal_age_float'] + (df['days_in_shelter'] / 365)
print(df[['animal_age_float', 'days_in_shelter', 'age_at_outcome']].head(10))

#recalucate average days in shelter after fixing negative days
average_days_in_shelter = df.groupby('id')['days_in_shelter'].first().mean()
print(f'Average days in shelter: {average_days_in_shelter:.2f}')

#recalcualte min and max days in shelter after fixing negative days for sanity check
min_days = df.groupby('id')['days_in_shelter'].first().min()
max_days = df.groupby('id')['days_in_shelter'].first().max()
print(f'minimum days in shelter: {min_days}')
print(f'maximum days in shelter: {max_days}')

#new broken minimum, need to repeat the process of finding the problematic negative minimum value and print the corresponding rows
negative_days = df[df['days_in_shelter'] < 0]
print(negative_days[['id', 'intakedate', 'movementdate', 'days_in_shelter']])

#making sure i didnt delete anything
print(df.info())

#return location of max_days and associated columns
max_days_row = df[df['days_in_shelter'] == max_days]
print(max_days_row[['id', 'intakedate', 'movementdate', 'age_at_outcome', 'days_in_shelter']])

#make variable with unique species and their counts
unique_species= (df['speciesname'].value_counts()) 

#find average length of stay for each species
species_avg_days = df.groupby('speciesname')['days_in_shelter'].mean()
print(species_avg_days)

colour_kitties= df[df['speciesname'] == 'Cat']['basecolour'].value_counts()
print(colour_kitties)

colour_kitties_avg_days = df[df['speciesname'] == 'Cat'].groupby('basecolour')['days_in_shelter'].mean()
print(colour_kitties_avg_days)


