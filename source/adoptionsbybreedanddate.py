import pandas as pd
df = pd.read_csv('datasets/animal-data-1.csv')
print(df.head())
print (df.info())

animal_age=df['animalage']
print(animal_age.head())

#code from google ai and claude
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


df=df.drop(columns=['index','animalage','istransfer','sheltercode','identichipnumber','animalname','location','istrial','returndate','returnedreason','deceaseddate','deceasedreason','diedoffshelter','isdoa'])
print(list(df.columns))

