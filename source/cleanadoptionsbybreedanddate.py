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
#replace movement type values of Stolen, Escaped, and Released To Wild to Other
df['movementtype'] = df['movementtype'].replace(['Stolen', 'Escaped', 'Released To Wild'], 'Other')
#replace reclaimed to Return to Owner
df['movementtype'] = df["movementtype"].replace(['Reclaimed'], 'Return to Owner')
# outcome_type: ['Return to Owner', 'Transfer', 'Foster','Euthanasia', 'Adoption', 'Other']
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

#df.to_csv(r'datasets/shorttermABD.csv', index=False)

#print size of df
print (len(df))




import os
import plotting
#os.makedirs('figures', exist_ok=True)
#mi_df = plotting.mutual_info_regression_matrix(ogdf.dropna())
#plotting.plot_mutual_info_heatmap(mi_df, save_path='figures/ogdfABD_MI_heatmap.png')

#print all dog breeds
pd.set_option('display.max_columns', None)
print(df[df['animal_species'] == 'dog']['breed'].value_counts())

print(df[df['animal_species']=='cat']['breed'].value_counts())

#split dataset into cat and dog
pd.set_option('display.max_rows', None)
df_cat, df_dog= data_utils.split_cat_dog(df)

#df_cat.to_csv(r'datasets/newcatshorttermABD.csv', index=False)
#df_dog.to_csv(r'datasets/dogshorttermABD.csv', index=False)

#print(df_cat.head())

#names will be: DSH, DLH, DMH, 

print(df_cat['breed'].unique().sum)

df_cat['breed']= df_cat['breed'].replace(['Domestic Short Hair','Munchkin','British Shorthair/Domestic Short Hair','Egyptian Mau/Domestic Short Hair','DMH/DSH', 'DSH/Unknown', 'Calico', 'DSH', 'Exotic Shorthair/Extra-Toes Cat (Hemingway Polydactyl)', ],'American Shorthair')
df_cat['breed']=df_cat['breed'].replace(['Domestic Long Hair','Norwegian Forest Cat', 'Himalayan','DLH/Unknown','DMH/DLH', 'DLH'],'American Longhair')
df_cat['breed']=df_cat['breed'].replace(['Russian Blue'],'Domestic Short Hair/Russian Blue')
df_cat['breed']=df_cat['breed'].replace(['Domestic Medium Hair','Domestic Long Hair/Domestic Short Hair',  'Domestic Short Hair/DLH','DMH'], 'American Longhair')
df_cat['breed']=df_cat['breed'].replace(['British Shorthair/Unknown', 'British Shorthair/Mix', ],'British Shorthair')
df_cat['breed']=df_cat['breed'].replace(['Siamese/Mix', 'Siamese/Egyptian Mau', 'Siamese/Balinese','Domestic Medium Hair/Siamese','Siamese/DSH','Domestic Short Hair/Siamese', 'Balinese/Domestic Medium Hair', 'Domestic Long Hair/Siamese', 'Siamese/Domestic Short Hair'], 'Siamese Cat')
df_cat['breed']=df_cat['breed'].replace(['Siamese/Snowshoe', 'Domestic Short Hair/Snowshoe', 'Snowshoe/Domestic Short Hair', ], 'Snowshoe')
df_cat['breed']=df_cat['breed'].replace(['Manx/DSH','Manx/Domestic Short Hair', 'Domestic Short Hair/Manx','Domestic Long Hair/Manx', 'Domestic Medium Hair/Manx', 'Snowshoe/Mix','Siamese/Domestic Long Hair', 'Siamese/Manx', 'Balinese/Manx','Manx/Mix','Manx/Domestic Long Hair',], 'Manx')
df_cat['breed']=df_cat['breed'].replace(['Persian/DMH', 'Domestic Long Hair/Persian', 'Turkish Angora/Persian'], 'Persian')
df_cat['breed']=df_cat['breed'].replace(['American Curl/DSH', 'American Curl'], 'American Shorthair')
df_cat['breed']=df_cat['breed'].replace(['Domestic Long Hair/Maine Coon', 'Maine Coon/Domestic Long Hair','Maine Coon/Mix',], 'Maine Coon')
df_cat['breed']=df_cat['breed'].replace(['Oriental Shorthair/Domestic Short Hair', 'Oriental Shorthair', 'Oriental'], 'Oriental Bicolor')
df_cat['breed']=df_cat['breed'].replace(['Domestic Short Hair/Bengal', 'Bengal'], 'Bengal Cats')
df_cat['breed']=df_cat['breed'].replace(['Siamese/Ragdoll','Ragamuffin/Domestic Long Hair',], 'Ragdoll Cats')



#TODO snowshoe gets its own cateogry, siamese needs to be siamese cat, 

bloompop= pd.read_csv('datasets/bloompop.csv')

print(bloompop)
#bloompop.columns = ["year", "population"]

#populate 'population' column based on the year in intake_year with the values from bloompop.csv

#df_cat['population']= None



# Dictionary containing your rules
# year_map = {2009:71848, 2012:82208, 2013:82524, 2015:83698, 2016:84481, 2017:84945, 2018:85228, 2019:85610}

# # Populate the new column
# df_cat['population'] = df['year'].map(year_map)

bloompop['year'] = bloompop['year'].astype(int)
df_cat['intake_year'] = df_cat['intake_year'].astype(int)
df_cat = df_cat.merge(bloompop[["year", "population"]], left_on="intake_year", right_on="year", how='left')
df_dog['intake_year'] = df_dog['intake_year'].astype(int)
df_dog=df_dog.merge(bloompop[['year', 'population']], left_on='intake_year', right_on='year', how='left')

#year_map_m = bloompop.set_index("year")["pop"]
#df_cat["population"] = df_cat["intake_year"].map(year_map_m).fillna(df_cat["population"])
print(df_cat[['year','population']].head(15))
print(df_dog[['year', 'population']].head(15))

print(df_dog['intake_year'].unique())
print(bloompop['year'].unique())

df_dog.to_csv(r'datasets/dogshorttermABD.csv', index=False)
#df_cat.to_csv(r'datasets/newcatshorttermABD.csv', index=False)




