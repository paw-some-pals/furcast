import pandas as pd
import re
import numpy as np
from data_utils import categorize_color, categorize_breed_by_species, categorize_size,get_seasons
from plotting import mutual_info_regression_matrix, plot_mutual_info_heatmap

def reading_files():
    shelter_df = pd.read_csv("datasets/acc_cat_cleaned.csv")
    cat_traits = pd.read_csv("datasets/cat_breeds.csv")

    return shelter_df,cat_traits

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


def breed_mapping(shelter_df,cat_traits):
    MANUAL_BREED_MAP = {
    "Abyssinian":"Abyssinian",
    "American Curl Shorthair":"American Shorthair",
    "American Shorthair" : "American Shorthair",
    "American Wirehair":"American Wirehair",
    "Angora": "Turkish Angora",
    "Balinese": "Siamese Cat",
    "Bengal": "Bengal Cats",
    "Birman":"Birman",
    "Bombay":"Bombay",
    "British Shorthair": "British Shorthair",
    "Burmese": "Burmese",
    "Chartreux":"Russian Blue",
    "Cornish Rex":"Cornish Rex",
    "Cymric":"Manx",
    "Devon Rex":"Devon Rex",
    "Domestic Longhair":"American Longhair",
    "Domestic Medium Hair":"American Longhair",
    "Domestic Shorthair":"American Shorthair",
    "Exotic Shorthair":"Persian",
    "Himalayan":"Ragdoll Cats",
    "Japanese Bobtail":"Japanese Bobtail",
    "Javanese":"Javanese",
    "Maine Coon":"Maine Coon",
    "Manx":"Manx",
    "Munchkin Longhair":"American Longhair",
    "Munchkin Shorthair":"American Shorthair",
    "Norwegian Forest Cat":"American Longhair",
    "Ocicat":"American Shorthair",
    "Oriental Sh":"Oriental Bicolor",
    "Persian":"Persian",
    "Pixiebob Shorthair":"American Shorthair",
    "Ragdoll":"Ragdoll Cats",
    "Rex":"Devon Rex",
    "Russian Blue":"Russian Blue",
    "Scottish Fold":"Scottish Fold",
    "Siamese":"Siamese Cat",
    "Snowshoe":"Snowshoe",
    "Sphynx":"Sphynx",
    "Tonkinese":"Tonkinese",
    "Turkish Angora":"Turkish Angora",
    "Turkish Van":"Turkish Van"
    }


    # Temporary raw split columns
    shelter_df[["raw_breed_1", "raw_breed_2", "is_mixed"]] = shelter_df["breed"].apply(parse_raw_breed)

    shelter_df["breed_1"] = shelter_df["raw_breed_1"].map(MANUAL_BREED_MAP)
    shelter_df["breed_2"] = shelter_df["raw_breed_2"].map(MANUAL_BREED_MAP).fillna("None")

    # Drop temporary raw split columns
    shelter_df = shelter_df.drop(columns=["raw_breed_1", "raw_breed_2"])

    # Save files
    shelter_df.to_csv("datasets/acc_cat_breed_split.csv", index=False)

    print(f"Raw shelter breed strings: {shelter_df['breed'].nunique()}")

    return shelter_df



def create_heatmap(df_heatmap):
    mi_df = mutual_info_regression_matrix(df_heatmap, filename="figures/aac_cat_cleaned_heatmap.png")

def fill_values(df_kaggle,df):
    
    #df is cat breed split 
    
    kaggle_feature_cols = [c for c in df_kaggle.columns if c != 'name']

    pure_mask = df['breed_2'] == 'None'

    # Pure breeds: merge directly on breed_1
    df_pure = df[pure_mask].merge(
        df_kaggle, how='left', left_on='breed_1', right_on='name'
    ).drop(columns=['name'])

    # Mixed breeds: average kaggle features for breed_1 and breed_2
    df_mixed = df[~pure_mask].copy()

    b1_vals = df_mixed[['breed_1']].merge(
        df_kaggle, how='left', left_on='breed_1', right_on='name'
    )[kaggle_feature_cols].to_numpy()

    b2_vals = df_mixed[['breed_2']].merge(
        df_kaggle, how='left', left_on='breed_2', right_on='name'
    )[kaggle_feature_cols].to_numpy()

    df_mixed[kaggle_feature_cols] = np.nanmean([b1_vals, b2_vals], axis=0)

    return pd.concat([df_pure, df_mixed]).sort_index().reset_index(drop=True)

def adding_fetaures(cat_traits, split_df):
    cat_traits_simplified = cat_traits.drop(columns=["length","origin"])
    final_austin_cat_df = fill_values(cat_traits_simplified, split_df)


    return final_austin_cat_df


def main():
    shelter_df, cat_traits = reading_files()
    shelter_df_split = breed_mapping(shelter_df, cat_traits)
    final_austin_cat_df = adding_fetaures(cat_traits,shelter_df_split)
    final_austin_cat_df["season"] = final_austin_cat_df["intake_month"].apply(get_seasons)
    create_heatmap(final_austin_cat_df)
    final_austin_cat_df.to_csv("datasets/acc_cat_final.csv", index=False)


if __name__ == "__main__":
    main()