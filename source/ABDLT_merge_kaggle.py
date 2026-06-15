#assumes that cleaned dog and cat files are present as well as cat and dog features datasets from kaggle

import pandas as pd

def read_file():
    cat_df = pd.read_csv("datasets/ABDLT_output_cat.csv")
    dog_df = pd.read_csv("datasets/ABDLT_output_dog_matched.csv")
    dog_kaggle = pd.read_csv("datasets/dog_breeds.csv")
    cat_kaggle = pd.read_csv("datasets/cat_breeds.csv")

    return(cat_df, dog_df, dog_kaggle, cat_kaggle)

def clean_kaggle_dog(df):
    df = df.drop(columns=['min_life_expectancy', 'max_life_expectancy', 'max_height_male', 'max_height_female', 'max_weight_male', 'max_weight_female', 'min_height_male', 'min_height_female', 'min_weight_male', 'min_weight_female'])
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

cat_df, dog_df, dog_kaggle, cat_kaggle = read_file()
dog_kaggle = clean_kaggle_dog(dog_kaggle)
clean_dog = fill_values_dog(dog_df, dog_kaggle)


print(clean_dog.head(15))
#print(clean_cat.head(15))