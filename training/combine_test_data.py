import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AAC_cat = pd.read_csv("datasets/final/final_df_aac_cats.csv")
ADB_cat = pd.read_csv("datasets/final/adb_cat_nn.csv")

AAC_dog = pd.read_csv("datasets/final/final_df_aac_dogs.csv")
ADB_dog = pd.read_csv("datasets/final/adb_dog_nn.csv")


FEATURE_COLS_DOGS = [
    "age_intake",
    "sex",
    "spay_neuter",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    "animal_size",
    "colour",
    "intake_condition",
    "intake_type",
    "is_mixed",
    "breed_1",
    "breed_2",
    "good_with_children",
    "good_with_other_dogs",
    "shedding",
    "grooming",
    "drooling",
    "coat_length",
    "good_with_strangers",
    "playfulness",
    "protectiveness",
    "trainability",
    "energy",
    "barking",
    "season",
    "population",
    "unemploy_rate"
]

FEATURE_COLS_CATS = [
    "age_intake",
    "sex",
    "spay_neuter",
    "intake_month",
    "intake_day",
    "intake_year",
    "animal_species",
    "colour",
    "intake_condition",
    "intake_type",
    "is_mixed",
    "breed_1",
    "breed_2",
    "min_life_expectancy",
    "max_life_expectancy",
    "min_weight",
    "max_weight",
    "family_friendly",
    "shedding",
    "general_health",
    "playfulness",
    "children_friendly",
    "grooming",
    "intelligence",
    "other_pets_friendly",
    "black",
    "white",
    "season",
    "population",
    "unemploy_rate"
]

TARGET_CLF = "stay_category"


def split_three_way_n(df, feature_cols, n_val, n_test, random_state=34):
    """Split df into train/val/test using absolute row counts for val and test."""
    X = df[feature_cols]
    y = df[TARGET_CLF]

    test_frac = n_test / len(df)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state
    )

    val_frac = n_val / len(X_trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ADB datasets don't have spay_neuter or intake_condition — fill with placeholders
ADB_cat["spay_neuter"] = "Unknown"
ADB_cat["intake_condition"] = "Other"
ADB_dog["spay_neuter"] = "Unknown"
ADB_dog["intake_condition"] = "Other"


AAC_cat_X_train, AAC_cat_X_val, AAC_cat_X_test, AAC_cat_y_train, AAC_cat_y_val, AAC_cat_y_test = split_three_way_n(AAC_cat, FEATURE_COLS_CATS, n_val=400, n_test=400)
ADB_cat_X_train, ADB_cat_X_val, ADB_cat_X_test, ADB_cat_y_train, ADB_cat_y_val, ADB_cat_y_test = split_three_way_n(ADB_cat, FEATURE_COLS_CATS, n_val=400, n_test=400)

AAC_dog_X_train, AAC_dog_X_val, AAC_dog_X_test, AAC_dog_y_train, AAC_dog_y_val, AAC_dog_y_test = split_three_way_n(AAC_dog, FEATURE_COLS_DOGS, n_val=450, n_test=450)
ADB_dog_X_train, ADB_dog_X_val, ADB_dog_X_test, ADB_dog_y_train, ADB_dog_y_val, ADB_dog_y_test = split_three_way_n(ADB_dog, FEATURE_COLS_DOGS, n_val=450, n_test=450)

print(f"Cat val  — AAC: {len(AAC_cat_X_val)}, ADB: {len(ADB_cat_X_val)}")
print(f"Cat test — AAC: {len(AAC_cat_X_test)}, ADB: {len(ADB_cat_X_test)}")
print(f"Cat train— AAC: {len(AAC_cat_X_train)}, ADB: {len(ADB_cat_X_train)}")
print(f"Dog val  — AAC: {len(AAC_dog_X_val)}, ADB: {len(ADB_dog_X_val)}")
print(f"Dog test — AAC: {len(AAC_dog_X_test)}, ADB: {len(ADB_dog_X_test)}")
print(f"Dog train— AAC: {len(AAC_dog_X_train)}, ADB: {len(ADB_dog_X_train)}")


def make_full_df(X, y):
    df = X.copy()
    df[TARGET_CLF] = y.values
    return df


# Per-source train sets
cat_train_aac = make_full_df(AAC_cat_X_train, AAC_cat_y_train)
cat_train_adb = make_full_df(ADB_cat_X_train, ADB_cat_y_train)
dog_train_aac = make_full_df(AAC_dog_X_train, AAC_dog_y_train)
dog_train_adb = make_full_df(ADB_dog_X_train, ADB_dog_y_train)

# Combined val and test sets (shuffled)
combined_cat_val = make_full_df(
    pd.concat([AAC_cat_X_val, ADB_cat_X_val], ignore_index=True),
    pd.concat([AAC_cat_y_val, ADB_cat_y_val], ignore_index=True)
).sample(frac=1, random_state=34).reset_index(drop=True)

combined_cat_test = make_full_df(
    pd.concat([AAC_cat_X_test, ADB_cat_X_test], ignore_index=True),
    pd.concat([AAC_cat_y_test, ADB_cat_y_test], ignore_index=True)
).sample(frac=1, random_state=34).reset_index(drop=True)

combined_dog_val = make_full_df(
    pd.concat([AAC_dog_X_val, ADB_dog_X_val], ignore_index=True),
    pd.concat([AAC_dog_y_val, ADB_dog_y_val], ignore_index=True)
).sample(frac=1, random_state=34).reset_index(drop=True)

combined_dog_test = make_full_df(
    pd.concat([AAC_dog_X_test, ADB_dog_X_test], ignore_index=True),
    pd.concat([AAC_dog_y_test, ADB_dog_y_test], ignore_index=True)
).sample(frac=1, random_state=34).reset_index(drop=True)


OUT = "datasets/ensemble_part2"

# cat_train_aac.to_csv(f"{OUT}/cat_train_aac.csv", index=False)
# cat_train_adb.to_csv(f"{OUT}/cat_train_adb.csv", index=False)
# dog_train_aac.to_csv(f"{OUT}/dog_train_aac.csv", index=False)
# dog_train_adb.to_csv(f"{OUT}/dog_train_adb.csv", index=False)
combined_cat_val.to_csv(f"{OUT}/combined_cat_val.csv", index=False)
combined_cat_test.to_csv(f"{OUT}/combined_cat_test.csv", index=False)
combined_dog_val.to_csv(f"{OUT}/combined_dog_val.csv", index=False)
combined_dog_test.to_csv(f"{OUT}/combined_dog_test.csv", index=False)

print("done — saved to", OUT)