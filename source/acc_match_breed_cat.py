import pandas as pd
import re


shelter_df = pd.read_csv("datasets/acc_dog_cleaned.csv")
dog_traits = pd.read_csv("datasets/dog_breeds.csv")

# MANUAL_BREED_MAP = {
# "Abyssinian":"Abyssinian",
# "American Curl Shorthair":"American Shorthair",
# "American Shorthair" : "American Shorthair",
# "American Wirehair":"American Wirehair",
# "Angora": "Turkish Angora",
# "Balinese": ""
# "Bengal": "Bengal Cats"
# "Birman,Birman,100.0
# Bombay,Bombay,100.0
# British Shorthair,British Shorthair,100.0
# Burmese,Burmese,100.0
# Chartreux,,50.0
# Cornish Rex,Cornish Rex,100.0
# Cymric,,50.0
# Devon Rex,Devon Rex,100.0
# Domestic Longhair,British Longhair,66.66666666666667
# Domestic Medium Hair,,48.64864864864865
# Domestic Shorthair,American Shorthair,77.77777777777779
# Exotic Shorthair,British Shorthair,72.72727272727273
# Himalayan,,53.333333333333336
# Japanese Bobtail,Japanese Bobtail,100.0
# Javanese,Javanese,100.0
# Maine Coon,Maine Coon,100.0
# Manx,Manx,100.0
# Munchkin Longhair,,48.484848484848484
# Munchkin Shorthair,American Shorthair,66.66666666666667
# Norwegian Forest Cat,,45.16129032258065
# Ocicat,,36.36363636363637
# Oriental Sh,,59.25925925925925
# Persian,Persian,100.0
# Pixiebob Shorthair,British Shorthair,68.57142857142857
# Ragdoll,Ragdoll Cats,73.6842105263158
# Rex,,50.0
# Russian Blue,Russian Blue,100.0
# Scottish Fold,Scottish Fold,100.0
# Siamese,Siamese Cat,77.77777777777779
# Snowshoe,Snowshoe,100.0
# Sphynx,Sphynx,100.0
# Tonkinese,Tonkinese,100.0
# Turkish Angora,Turkish Angora,100.0
# Turkish Van,Turkish Van,100.0

# }


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


# Temporary raw split columns
shelter_df[["raw_breed_1", "raw_breed_2", "is_mixed"]] = shelter_df["breed"].apply(parse_raw_breed)



shelter_df["breed_1"] = shelter_df["raw_breed_1"].map(MANUAL_BREED_MAP)
shelter_df["breed_2"] = shelter_df["raw_breed_2"].map(MANUAL_BREED_MAP).fillna("None")

# Drop temporary raw split columns
shelter_df = shelter_df.drop(columns=["raw_breed_1", "raw_breed_2"])

# Save files
shelter_df.to_csv("datasets/acc_cat_breed_split.csv", index=False)

print(f"Raw shelter breed strings: {shelter_df['breed'].nunique()}")