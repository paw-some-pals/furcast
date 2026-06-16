import os

import pandas as pd
from sklearn.neighbors import NearestNeighbors


def compute_nn(
    neighbours_db: pd.DataFrame,
    samples: pd.DataFrame,
    target_column: str,
    similarity_metric: str = "cosine",
):
    """
    For each row in samples, find its nearest neighbour in neighbours_db,
    then copy that neighbour's target_column value into the sample.

    neighbours_db:
        DataFrame containing labelled neighbour examples.

    samples:
        DataFrame containing samples whose target_column should be filled.

    target_column:
        Column whose value should be copied from the nearest neighbour.

    similarity_metric:
        Metric passed to sklearn.neighbors.NearestNeighbors.
        Examples: "cosine", "euclidean", "manhattan".

    Returns:
        A new DataFrame with the same rows as samples, but target_column
        replaced by the target value of the nearest neighbour.
    """

    if target_column not in neighbours_db.columns:
        raise ValueError(f"{target_column=} is not in neighbours_db")

    if target_column not in samples.columns:
        raise ValueError(f"{target_column=} is not in samples")

    # Use shared numeric columns, excluding the target column, as features.
    feature_columns = [
        col for col in samples.columns
        if (
            col in neighbours_db.columns
            and col != target_column
            and pd.api.types.is_numeric_dtype(samples[col])
            and pd.api.types.is_numeric_dtype(neighbours_db[col])
        )
    ]

    if len(feature_columns) == 0:
        raise ValueError(
            "No shared numeric feature columns found between samples and neighbours_db."
        )

    X_neighbours = neighbours_db[feature_columns]
    X_samples = samples[feature_columns]

    if X_neighbours.isna().any().any():
        raise ValueError("neighbours_db contains NaNs in the feature columns.")

    if X_samples.isna().any().any():
        raise ValueError("samples contains NaNs in the feature columns.")

    # Fit nearest-neighbour model on the database.
    nn = NearestNeighbors(
        n_neighbors=1,
        metric=similarity_metric,
    )

    nn.fit(X_neighbours)

    # indices[i, 0] is the index-position in neighbours_db of the nearest
    # neighbour for samples.iloc[i].
    distances, indices = nn.kneighbors(X_samples)

    nearest_targets = neighbours_db.iloc[indices[:, 0]][target_column].to_numpy()

    output = samples.copy()
    output[target_column] = nearest_targets

    return output


if __name__ == "__main__":
    out_dir = "datasets/final"
    os.makedirs(out_dir, exist_ok=True)

    aac_cat = pd.read_csv("datasets/final/final_df_aac_cats.csv")
    aac_dog = pd.read_csv("datasets/final/final_df_aac_dogs.csv")
    adb_cat = pd.read_csv("datasets/final/ABDST_output_cat_pop.csv")
    adb_dog = pd.read_csv("datasets/final/ABDST_output_dog_pop.csv")

    adb_dog["intake_condition"] = ""
    adb_cat["intake_condition"] = ""
    adb_dog["spay_neuter"] = ""
    adb_cat["spay_neuter"] = ""

    for species, neighbours, samples in [("cat", aac_cat, adb_cat), ("dog", aac_dog, adb_dog)]:
        for target in ["intake_condition", "spay_neuter"]:
            samples[target] = compute_nn(neighbours, samples, target_column=target)[target]
        out_path = os.path.join(out_dir, f"adb_{species}_nn.csv")
        samples.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

