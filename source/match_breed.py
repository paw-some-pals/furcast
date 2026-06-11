import pandas as pd
import re
from rapidfuzz import process, fuzz

shelter_df = pd.read_csv("datasets/acc_dog_cleaned.csv")
dog_traits = pd.read_csv("datasets/dog_breeds.csv")

KAGGLE_BREED_COL = "Name"
MATCH_THRESHOLD = 60

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

# Map matched Kaggle names back into main df
breed_map = breed_lookup.set_index("raw_breed")["kaggle_breed"].to_dict()

shelter_df["breed_1"] = shelter_df["raw_breed_1"].map(breed_map)
shelter_df["breed_2"] = shelter_df["raw_breed_2"].map(breed_map)

# Breeds that did NOT match Kaggle at threshold
breeds_for_gemini = (
    breed_lookup[breed_lookup["kaggle_breed"].isna()][["raw_breed"]]
    .rename(columns={"raw_breed": "breed"})
    .sort_values("breed")
    .reset_index(drop=True)
)

# Drop temporary raw split columns
shelter_df = shelter_df.drop(columns=["raw_breed_1", "raw_breed_2"])

# Save files
shelter_df.to_csv("datasets/acc_dog_breed_split.csv", index=False)
breeds_for_gemini.to_csv("datasets/breeds_for_gemini.csv", index=False)
breed_lookup.to_csv("datasets/breed_match_audit.csv", index=False)

print(f"Raw shelter breed strings: {shelter_df['breed'].nunique()}")
print(f"Unique parsed raw breeds: {len(breed_lookup)}")
print(f"Matched to Kaggle at {MATCH_THRESHOLD}%+: {breed_lookup['kaggle_breed'].notna().sum()}")
print(f"Breeds needing Gemini: {len(breeds_for_gemini)}")
print(breeds_for_gemini.head(30))