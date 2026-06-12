import pandas as pd
import re
from rapidfuzz import process, fuzz

shelter_df = pd.read_csv("datasets/ABDLT_output_dog.csv")
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
shelter_df.to_csv("datasets/acc_dog_breed_split.csv", index=False)
breeds_for_gemini.to_csv("datasets/breeds_for_gemini.csv", index=False)
breed_lookup.to_csv("datasets/breed_match_audit.csv", index=False)

print(f"Raw shelter breed strings: {shelter_df['breed'].nunique()}")
print(f"Unique parsed raw breeds: {len(breed_lookup)}")
print(f"Matched to Kaggle at {MATCH_THRESHOLD}%+: {breed_lookup['kaggle_breed'].notna().sum()}")
print(f"Breeds needing Gemini: {len(breeds_for_gemini)}")
print(breeds_for_gemini.head(30))


print(dog_traits['Name'].unique())