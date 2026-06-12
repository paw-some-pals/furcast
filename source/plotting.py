import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


from sklearn.feature_selection import mutual_info_regression


def mutual_info_regression_matrix(
    df,
    *,
    filename="figures/MI_heatmap.png",
    discrete_features="auto",
    n_neighbors=3,
    random_state=0,
    symmetrize=True,
    fill_diag=0.0,
):
    """
    Pairwise MI matrix for mostly continuous-valued features.
    """

    df_num = df.copy()

    for col in df_num.columns:
        if not pd.api.types.is_numeric_dtype(df_num[col]):
            df_num[col] = LabelEncoder().fit_transform(df_num[col].astype(str))

    cols = df_num.columns
    n = len(cols)
    mi = np.zeros((n, n), dtype=float)

    X_all = df_num.to_numpy()

    for j, target_col in enumerate(cols):
        y = df_num[target_col].to_numpy()

        scores = mutual_info_regression(
            X_all,
            y,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )

        mi[:, j] = scores

    if symmetrize:
        mi = 0.5 * (mi + mi.T)

    np.fill_diagonal(mi, fill_diag)

    mutal_info_df = pd.DataFrame(mi, index=cols, columns=cols)
    plot_mutual_info_heatmap(mutal_info_df, save_path=filename)

    return pd.DataFrame(mi, index=cols, columns=cols)

# added save_path argument so does not overwrite the same file when run for our different datasets
def plot_mutual_info_heatmap(mi_df, figsize=(10, 8), annot=True, save_path="figures/MI_heatmap.png"):
    plt.figure(figsize=figsize)
    sns.heatmap(
        mi_df,
        cmap="viridis",
        annot=annot,
        fmt=".3f",
        square=True,
        cbar_kws={"label": "Mutual information"},
    )
    plt.title("Pairwise Mutual Information Heatmap between Continuous Features")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

# add so it doesnt run when imported as a module, but only when run directly
if __name__ == "__main__":
    # ------------------- USAGE EXAMPLE ------------------
    rng = np.random.default_rng(0)
    n = 500

    x = rng.normal(size=n)

    toy_df = pd.DataFrame({
        # Base continuous feature
        "x": x,

        # Strong linear relationship with x
        "linear_x": 2.0 * x + 0.2 * rng.normal(size=n),

        # Nonlinear relationship with x
        "quadratic_x": x**2 + 0.2 * rng.normal(size=n),

        # Periodic nonlinear relationship with x
        "sin_x": np.sin(3 * x) + 0.1 * rng.normal(size=n),

        # Mostly independent noise
        "noise": rng.normal(size=n),

        # Discretized/binned version of x
        "x_bin": pd.qcut(x, q=4, labels=False),

        # Another categorical-ish variable partially dependent on x
        "category_like": np.where(x > 0.5, 2, np.where(x < -0.5, 0, 1)),
    })

    toy_df.head()
    mi_df = mutual_info_regression_matrix(toy_df)
    plot_mutual_info_heatmap(mi_df, annot=True)