#add season

from data_utils import get_seasons
import pandas as pd
import numpy as np

df_STcat_final= pd.read_csv('datasets/newcatshorttermABD.csv')
df_STdog_final= pd.read_csv('datasets/dogshorttermABD.csv')

def apply_get_season(df):
    df['season']=df['intake_month'].apply(get_seasons)
    return df

df_STcat_final= apply_get_season(df_STcat_final)

df_STdog_final= apply_get_season(df_STdog_final)

print(df_STcat_final[['intake_month', 'season']])

#add unemployment
def add_unemployment(df):
    unemp = pd.read_csv("datasets/blooming_unemploy.csv")
    unemp["intake_year"] = unemp["Year"].astype(int)
    unemp["intake_month"] = pd.to_datetime(unemp["Period"], format="%b").dt.month
    unemp["unemploy_rate"] = pd.to_numeric(
        unemp["unemployment rate"].str.replace(r"\([^)]+\)", "", regex=True).str.strip(),
        errors='coerce'
    )
    unemp = unemp[["intake_year", "intake_month", "unemploy_rate"]]

    return df.merge(unemp, how="left", on=["intake_year", "intake_month"])

df_STcat_final= add_unemployment(df_STcat_final)
df_STdog_final= add_unemployment(df_STdog_final)
df_STdog_final.to_csv(r'datasets/STdogADB.csv')
df_STcat_final.to_csv(r'datasets/STcatADB.csv')

print(df_STdog_final[['unemploy_rate', 'season']])

#do breed mapping
import pandas as pd
import re
from rapidfuzz import process, fuzz

shelter_df = pd.read_csv("datasets/STdogADB.csv")
dog_traits = pd.read_csv("datasets/dog_breeds.csv")

KAGGLE_BREED_COL = "Name"
MATCH_THRESHOLD = 60
MANUAL_BREED_MAP = {
    "Akbash": "Great Pyrenees",
    "Belgian Tervuren": "Belgian Malinois",
    "Black": "Labrador Retriever",  # generic fallback
    "Black Mouth Cur": "American Leopard Hound",
    "Blue Lacy": "Australian Cattle Dog",
    "Bouv Flandres": "Giant Schnauzer",
    "Boykin Span": "American Water Spaniel",
    "Bruss Griffon": "Pug",
    "Canaan Dog": "Jindo",
    "Carolina Dog": "Basenji",
    "Catahoula": "American Leopard Hound",
    "Chesa Bay Retr": "Labrador Retriever",
    "Cirneco": "Basenji",
    "Collie Rough": "Shetland Sheepdog",
    "Collie Smooth": "Shetland Sheepdog",
    "Dandie Dinmont": "Norfolk Terrier",
    "Dogo Argentino": "American Bulldog",
    "Dogue De Bordeaux": "Bullmastiff",
    "Eng Toy Spaniel": "Cavalier King Charles Spaniel",
    "English Foxhound": "Treeing Walker Coonhound",
    "English Pointer": "German Longhaired Pointer",
    "English Springer Spaniel": "Field Spaniel",
    "Entlebucher": "Australian Cattle Dog",
    "Feist": "American Hairless Terrier",
    "Finnish Spitz": "Shiba Inu",
    "German Shorthair Pointer": "German Longhaired Pointer",
    "Glen Of Imaal": "Norfolk Terrier",
    "Grand Basset Griffon Vendeen": "Basset Hound",
    "Greater Swiss Mountain Dog": "Bernese Mountain Dog",
    "Harrier": "Beagle",
    "Irish Wolfhound": "Great Dane",
    "Kangal": "Anatolian Shepherd Dog",
    "Kuvasz": "Great Pyrenees",
    "Landseer": "Newfoundland",
    "Leonberger": "Newfoundland",
    "Lhasa Apso": "Shih Tzu",
    "Lowchen": "Bichon Frise",
    "Norwegian Elkhound": "Keeshond",
    "Old English Sheepdog": "Komondor",
    "Patterdale Terr": "Norfolk Terrier",
    "Pbgv": "Basset Hound",  # Petit Basset Griffon Vendéen
    "Picardy Sheepdog": "Berger Picard",
    "Podengo Pequeno": "Basenji",
    "Pointer": "German Longhaired Pointer",
    "Port Water Dog": "Barbet",
    "Presa Canario": "Cane Corso",
    "Redbone Hound": "Plott Hound",
    "Saluki": "Afghan Hound",
    "Sealyham Terr": "West Highland White Terrier",
    "Soft Coated Wheaten Terrier": "Irish Terrier",
    "Spinone Italiano": "Bracco Italiano",
    "St. Bernard Rough Coat": "Bernese Mountain Dog",
    "St. Bernard Smooth Coat": "Bernese Mountain Dog",
    "Standard Schnauzer": "Giant Schnauzer",
    "Sussex Span": "Field Spaniel",
    "Swedish Vallhund": "Pembroke Welsh Corgi",
    "Swiss Hound": "Treeing Walker Coonhound",
    "Treeing Cur": "American Leopard Hound",
    "Treeing Tennesse Brindle": "Plott Hound",
    "Unknown": "Labrador Retriever",  # safest default
    "Weimaraner": "Vizsla",
    "Welsh Springer Spaniel": "Field Spaniel",
    "Wirehaired Pointing Griffon": "German Longhaired Pointer",
    "Mexican Hairless": "Xoloitzcuintli",
    "Queensland Heeler": "Australian Cattle Dog",
    "Standard Poodle": "Poodle (Miniature)",
    "Toy Poodle": "Poodle (Miniature)",
}

# breed_mapping = {
#     'Bouvier des Flanders': 'Bergamasco Sheepdog',
#     'Brussels Griffon/Havanese': 'Norfolk Terrier',
#     'Bully Breed': 'American Staffordshire Terrier',
#     'Bully Breed Mix': 'American Staffordshire Terrier',
#     'Labrador Retriever/Bully Breed Mix': 'American Staffordshire Terrier',
#     'Bully Breed Mix/Mastiff': 'American Staffordshire Terrier',
#     'Beagle/Bully Breed Mix': 'American Staffordshire Terrier',
#     'Boxer/Bully Breed Mix': 'American Staffordshire Terrier',


#     'Pitbull': 'American Staffordshire Terrier',
#     'Chesapeake Bay Retriever': 'Labrador Retriever',
#     'Curly-coated Retriever': 'Labrador Retriever',
#     'Cockapoo': 'Poodle (Miniature)',
#     'Poodle': 'Poodle (Miniature)',
#     'Poodle, Standard': 'Poodle (Miniature)',
#     'Poodle, Toy': 'Poodle (Miniature)',
#     'Collie, Smooth': 'Border Collie',
#     'Corgi': 'Pembroke Welsh Corgi',
#     'German Shorthaired Pointer': 'German Longhaired Pointer',
#     'Wire-haired Pointing Griffon': 'German Longhaired Pointer',
#     'Heeler': 'Australian Cattle Dog',
#     'Husky': 'Siberian Husky',
#     'Manchester Terrier, Toy': 'Miniature Pinscher',
#     'Saint Bernard': 'Newfoundland',
#     'Saint Bernard St. Bernard': 'Newfoundland',
#     'Schnauzer, Standard': 'Giant Schnauzer',
#     'Shar Pei': 'Chinese Shar-Pei',
#     'Sheltie': 'Shetland Sheepdog',
#     'Shep': 'Belgian Malinois',
#     'Shepherd': 'Belgian Malinois',
#     'Soft Coated Wheaten Terrier': 'Irish Terrier',
#     'Spitz': 'Keeshond',
#     'Springer Spaniel': 'Field Spaniel'
# }

#shelter_df['breed'] = shelter_df['breed'].replace(breed_mapping)

breed_substitutions = {
    'Bouvier des Flanders': 'Bergamasco Sheepdog',
    'Brussels Griffon': 'Norfolk Terrier',
    'Bully Breed': 'American Staffordshire Terrier',
    'Pitbull': 'American Staffordshire Terrier',
    'Chesapeake Bay Retriever': 'Labrador Retriever',
    'Cockapoo': 'Poodle (Miniature)',
    'Collie, Smooth': 'Border Collie',
    'Corgi': 'Pembroke Welsh Corgi',
    'Curly- coated Retriever': 'Labrador Retriever',
    'German Shorthaired Pointer': 'German Longhaired Pointer',
    'Wire-haired Pointing Griffon': 'German Longhaired Pointer',
    'Heeler': 'Australian Cattle Dog',
    'Husky': 'Siberian Husky',
    'Manchester Terrier, Toy': 'Miniature Pinscher',
    'Poodle, Standard': 'Poodle (Miniature)',
    'Poodle, Toy': 'Poodle (Miniature)',
    'Poodle': 'Poodle (Miniature)',
    'Saint Bernard St. Bernard': 'Newfoundland',
    'Saint Bernard': 'Newfoundland',
    'Schnauzer, Standard': 'Giant Schnauzer',
    'Shar Pei': 'Chinese Shar-Pei',
    'Sheltie': 'Shetland Sheepdog',
    'Soft Coated Wheaton Terrier': 'Irish Terrier',
    'Shep Mix': 'Belgian Malinois',
    'German Shepherd Dog': 'Belgian Malinois',
    'Belgian Shepherd': 'Belgian Malinois',
    'Shepherd': 'Belgian Malinois',
    'Soft Coated Wheaten Terrier': 'Irish Terrier',
    'Spitz': 'Keeshond',
    'Springer Spaniel': 'Field Spaniel',
    'English Springer Spaniel': 'Field Spaniel'
}

for old, new in breed_substitutions.items():
    shelter_df['breed'] = shelter_df['breed'].str.replace(
        old,
        new,
        regex=False
    )

KAGGLE_BREEDS = (
    dog_traits[KAGGLE_BREED_COL]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

def parse_raw_breed(breed):
    if pd.isna(breed):
        return pd.Series([None, None, False])

    raw = str(breed).strip()
    is_mixed = bool(re.search(r"\bMix\b|/", raw, flags=re.IGNORECASE))

    cleaned = re.sub(r"\bMix\b", "", raw, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in cleaned.split("/")]

    raw_breed_1 = parts[0] if len(parts) > 0 and parts[0] else None
    raw_breed_2 = parts[1] if len(parts) > 1 and parts[1] else None

    return pd.Series([raw_breed_1, raw_breed_2, is_mixed])

def match_to_kaggle(breed):
    if pd.isna(breed):
        return pd.Series([None, 0])

    result = process.extractOne(
        breed,
        KAGGLE_BREEDS,
        scorer=fuzz.token_sort_ratio
    )

    if result is None:
        return pd.Series([None, 0])

    match, score, _ = result

    if score >= MATCH_THRESHOLD:
        return pd.Series([match, score])

    return pd.Series([None, score])

# Temporary raw split columns
shelter_df[["raw_breed_1", "raw_breed_2", "is_mixed"]] = shelter_df["breed"].apply(parse_raw_breed)

# Build unique raw breed lookup
unique_raw_breeds = pd.concat([
    shelter_df["raw_breed_1"],
    shelter_df["raw_breed_2"]
]).dropna().astype(str).str.strip()

unique_raw_breeds = (
    unique_raw_breeds[unique_raw_breeds.ne("")]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

breed_lookup = pd.DataFrame({"raw_breed": unique_raw_breeds})

breed_lookup[["kaggle_breed", "match_score"]] = breed_lookup["raw_breed"].apply(
    match_to_kaggle
)

# Fill remaining NaN matches with manual mappings
breed_lookup["kaggle_breed"] = breed_lookup["kaggle_breed"].fillna(
    breed_lookup["raw_breed"].map(MANUAL_BREED_MAP)
)

# Map matched Kaggle names back into main df
breed_map = breed_lookup.set_index("raw_breed")["kaggle_breed"].to_dict()

shelter_df["breed_1"] = shelter_df["raw_breed_1"].map(breed_map)
shelter_df["breed_2"] = shelter_df["raw_breed_2"].map(breed_map).fillna("None")

# Breeds that did NOT match Kaggle at threshold or via manual mapping
breeds_for_gemini = (
    breed_lookup[breed_lookup["kaggle_breed"].isna()][["raw_breed"]]
    .rename(columns={"raw_breed": "breed"})
    .sort_values("breed")
    .reset_index(drop=True)
)

# Drop temporary raw split columns
shelter_df = shelter_df.drop(columns=["raw_breed_1", "raw_breed_2"])

# Save files
shelter_df.to_csv("datasets/ABDSTdog_output_dog_matched.csv", index=False)
breeds_for_gemini.to_csv("datasets/breeds_for_gemini.csv", index=False)
breed_lookup.to_csv("datasets/breed_match_audit.csv", index=False)

print(f"Raw shelter breed strings: {shelter_df['breed'].nunique()}")
print(f"Unique parsed raw breeds: {len(breed_lookup)}")
print(f"Matched to Kaggle at {MATCH_THRESHOLD}%+: {breed_lookup['kaggle_breed'].notna().sum()}")
print(f"Breeds needing Gemini: {len(breeds_for_gemini)}")
print(breeds_for_gemini.head(30))


#print(dog_traits['Name'].unique())
#merge kaggle 

#assumes that cleaned dog and cat files are present as well as cat and dog features datasets from kaggle
#assumes that source/cleanadoptionsbybreedanddatelongterm.py and source/match_breed_ABDLT_dog.py have been run 

import pandas as pd

def read_file():
    cat_df = pd.read_csv("datasets/STcatADB.csv")
    dog_df = pd.read_csv("datasets/ABDSTdog_output_dog_matched.csv")
    dog_kaggle = pd.read_csv("datasets/dog_breeds.csv")
    cat_kaggle = pd.read_csv("datasets/cat_breeds.csv")

    return(cat_df, dog_df, dog_kaggle, cat_kaggle)

def clean_kaggle_dog(df):
    df = df.drop(columns=['min_life_expectancy', 'max_life_expectancy', 'max_height_male', 'max_height_female', 'max_weight_male', 'max_weight_female', 'min_height_male', 'min_height_female', 'min_weight_male', 'min_weight_female'])
    return df

def clean_kaggle_cat(df):
    df = df.drop(columns=['length', 'origin', 'min_life_expectancy', 'max_life_expectancy', "min_weight", "max_weight"])
    return df

def fill_values_dog(df, df_kaggle):
    kaggle_feature_cols = [c for c in df_kaggle.columns if c != 'Name']

    pure_mask = df['breed_2'].isna() | (df['breed_2'] == 'None')

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

def fill_values_cat(df, df_kaggle):
    return df.merge(
        df_kaggle, how='left', left_on='breed', right_on='name'
    ).drop(columns=['name'])


#merge unemployment, population, season
def add_population(df):
    populations = pd.read_csv("datasets/blooming_population.csv")
    populations = populations[["Year", "Population"]].rename(
        columns={"Year": "intake_year", "Population": "population"}
    )
    populations["population"] = (
        populations["population"].astype(str).str.replace(",", "", regex=False).astype(int)
    )

    return df.merge(populations, how="left", on="intake_year")

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

def add_unemployment(df):
    unemp = pd.read_csv("datasets/blooming_unemploy.csv")
    unemp['Year'] = unemp['Year'].astype(int)
    unemp['Period'] = pd.to_datetime(unemp['Period'], format='%b').dt.month
    unemp["unemployment rate"] = pd.to_numeric(
        unemp["unemployment rate"].str.replace(r"%|\(.*?\)|-", "", regex=True).str.strip(),
        errors="coerce"
    )

    unemp = unemp[["Year", "Period", "unemployment rate"]]
    unemp.rename(columns={'Year': 'intake_year', 'Period': 'intake_month', 'unemployment rate': 'unemploy_rate'}, inplace=True)

    return df.merge(unemp, how="left", on=["intake_year", "intake_month"])


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


cat_df, dog_df, dog_kaggle, cat_kaggle = read_file()

#merge datasets: kaggle, unemployment, population and seasons to dog dataset 
dog_kaggle = clean_kaggle_dog(dog_kaggle)
merge_dog = fill_values_dog(dog_df, dog_kaggle)
#merge_dog = add_population(merge_dog)
#merge_dog = add_unemployment(merge_dog)
#merge_dog = apply_get_season(merge_dog)

#merge datasets: kaggle, unemployment, population and seasons to cat dataset 
cat_kaggle = clean_kaggle_cat(cat_kaggle)
merge_cat = fill_values_cat(cat_df, cat_kaggle)
# merge_cat = add_population(merge_cat)
# merge_cat = add_unemployment(merge_cat)
# merge_cat = apply_get_season(merge_cat)

merge_cat= apply_stay_category(merge_cat)
merge_dog= apply_stay_category(merge_dog)

merge_cat['is_mixed']= False
merge_cat=merge_cat.rename(columns={'breed': 'breed_1'})
merge_cat['breed_2']= 'None'

def check_colour(row):
    '''
    Usage df[['black', 'white']] = df.apply(check_colour, axis=1, result_type='expand')
    '''
    if row['colour'] == 'Black':
        return 1, 0  
    elif row['colour'] == 'White':
        return 0, 1
    else:
        return 0, 0

merge_cat[['black', 'white']]= merge_cat.apply(check_colour, axis=1, result_type='expand')

from data_utils import categorize_dog_breed_by_size

merge_dog["animal_size"] = merge_dog.apply(categorize_dog_breed_by_size, axis=1)

    
#print(merge_dog.head(15))
print(merge_cat.head(15))

merge_cat.to_csv("datasets/ABDST_output_cat_pop.csv", index=False)
merge_dog.to_csv("datasets/ABDST_output_dog_pop.csv", index=False)