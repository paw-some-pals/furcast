#assumes that cleaned dog and cat files are present as well as cat and dog features datasets from kaggle
#assumes that source/cleanadoptionsbybreedanddatelongterm.py and source/match_breed_ABDLT_dog.py have been run 

import pandas as pd
import plotting
import seaborn as sns

def read_file():
    cat_df = pd.read_csv("datasets/ABDLT_output_cat.csv")
    dog_df = pd.read_csv("datasets/ABDLT_output_dog_matched.csv")
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

cat_df, dog_df, dog_kaggle, cat_kaggle = read_file()

#merge datasets: kaggle, unemployment, population and seasons to dog dataset 
dog_kaggle = clean_kaggle_dog(dog_kaggle)
merge_dog = fill_values_dog(dog_df, dog_kaggle)
merge_dog = add_population(merge_dog)
merge_dog = add_unemployment(merge_dog)
merge_dog = apply_get_season(merge_dog)

#merge datasets: kaggle, unemployment, population and seasons to cat dataset 
cat_kaggle = clean_kaggle_cat(cat_kaggle)
merge_cat = fill_values_cat(cat_df, cat_kaggle)
merge_cat = add_population(merge_cat)
merge_cat = add_unemployment(merge_cat)
merge_cat = apply_get_season(merge_cat)

print(merge_dog.head(15))
print(merge_cat.head(15))

plotting.mutual_info_regression_matrix(merge_cat.dropna(),filename='figures/MI_heatmap_ABDLT_cat.png', figsize=(12,10))
plotting.mutual_info_regression_matrix(merge_dog.dropna(),filename='figures/MI_heatmap_ABDLT_dog.png', figsize=(14,12))

#merge_cat.to_csv("datasets/ABDLT_output_cat_pop.csv", index=False)
#merge_dog.to_csv("datasets/ABDLT_output_dog_pop.csv", index=False)