from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data" / "processed" / "phif_ml_dataset_v1.csv"

OUTPUT_DIR = ROOT / "reports" / "models"

IMPORTANCE_OUTPUT = (
    OUTPUT_DIR / "phif_hgb_permutation_importance.csv"
)

WELL_IMPORTANCE_OUTPUT = (
    OUTPUT_DIR / "phif_hgb_permutation_importance_by_well.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "GR",
    "LOG10_RT",
]

TARGET = "PHIF"

RANDOM_STATE = 42


# ============================================================
# HGB CONFIGURATION
# Same configuration as champion model
# ============================================================

HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": RANDOM_STATE,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HGB EXPLAINABILITY ANALYSIS")
print("=" * 80)

print("\nDataset:")
print(DATASET)

df = pd.read_csv(DATASET)

# ============================================================
# DERIVE MODEL FEATURE
# ============================================================
# LOG10_RT was used by the HGB model but is not stored
# directly in phif_ml_dataset_v1.csv.
#
# Recreate it deterministically from RT.
# This keeps the source dataset unchanged.
# ============================================================

if "LOG10_RT" not in df.columns:

    if "RT" not in df.columns:
        raise ValueError(
            "Dataset contains neither LOG10_RT nor RT."
        )

    # RT must be positive for log10 transformation.
    df["LOG10_RT"] = np.where(
        df["RT"] > 0,
        np.log10(df["RT"]),
        np.nan,
    )

    print(
        "\n[INFO] LOG10_RT was derived from RT using log10(RT)."
    )


required_columns = [
    "WELL",
    TARGET,
    *FEATURES,
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# Keep complete cases
df = df[
    df[required_columns]
    .notna()
    .all(axis=1)
].copy()


print("\nDataset shape:")
print(df.shape)

print("\nWells:")
print(
    df["WELL"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


# ============================================================
# STRICT LEAVE-ONE-WELL-OUT
# ============================================================

wells = sorted(
    df["WELL"]
    .drop_duplicates()
    .tolist()
)


all_results = []
fold_importance = []


for held_out_well in wells:

    print("\n" + "-" * 80)
    print(f"HELD-OUT WELL: {held_out_well}")
    print("-" * 80)

    train = df[
        df["WELL"] != held_out_well
    ].copy()

    test = df[
        df["WELL"] == held_out_well
    ].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # --------------------------------------------------------
    # Train HGB
    # --------------------------------------------------------

    model = HistGradientBoostingRegressor(
        **HGB_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Permutation importance on HELD-OUT WELL
    #
    # This is the important part:
    # feature importance is measured on unseen well data.
    # --------------------------------------------------------

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    mean_importance = result.importances_mean
    std_importance = result.importances_std

    for feature, importance, std in zip(
        FEATURES,
        mean_importance,
        std_importance,
    ):

        fold_importance.append(
            {
                "HELD_OUT_WELL": held_out_well,
                "FEATURE": feature,
                "PERMUTATION_IMPORTANCE": importance,
                "IMPORTANCE_STD": std,
                "N_TEST": len(test),
            }
        )

        print(
            f"{feature:12s} "
            f"importance={importance:.6f} "
            f"std={std:.6f}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    fold_importance
)


# ============================================================
# SAVE PER-WELL RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

importance_df.to_csv(
    WELL_IMPORTANCE_OUTPUT,
    index=False,
)


# ============================================================
# AGGREGATE FEATURE IMPORTANCE
# ============================================================

summary = (
    importance_df
    .groupby("FEATURE")
    .agg(
        MEAN_IMPORTANCE=(
            "PERMUTATION_IMPORTANCE",
            "mean",
        ),
        MEDIAN_IMPORTANCE=(
            "PERMUTATION_IMPORTANCE",
            "median",
        ),
        STD_ACROSS_WELLS=(
            "PERMUTATION_IMPORTANCE",
            "std",
        ),
        MEAN_IMPORTANCE_STD=(
            "IMPORTANCE_STD",
            "mean",
        ),
        N_WELLS=(
            "HELD_OUT_WELL",
            "nunique",
        ),
    )
    .reset_index()
)


# ============================================================
# FEATURE RANK
# ============================================================

summary = summary.sort_values(
    "MEAN_IMPORTANCE",
    ascending=False,
).reset_index(drop=True)

summary["FEATURE_RANK"] = (
    np.arange(len(summary)) + 1
)


# ============================================================
# POSITIVE / NEGATIVE / ZERO COUNTS
# ============================================================

sign_summary = (
    importance_df
    .assign(
        SIGN=lambda x: np.select(
            [
                x["PERMUTATION_IMPORTANCE"] > 0,
                x["PERMUTATION_IMPORTANCE"] < 0,
            ],
            [
                "POSITIVE",
                "NEGATIVE",
            ],
            default="ZERO",
        )
    )
    .groupby(
        ["FEATURE", "SIGN"]
    )
    .size()
    .unstack(
        fill_value=0
    )
    .reset_index()
)


summary = summary.merge(
    sign_summary,
    on="FEATURE",
    how="left",
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    IMPORTANCE_OUTPUT,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("AGGREGATED HELD-OUT-WELL PERMUTATION IMPORTANCE")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# PER-WELL MATRIX
# ============================================================

matrix = (
    importance_df
    .pivot(
        index="HELD_OUT_WELL",
        columns="FEATURE",
        values="PERMUTATION_IMPORTANCE",
    )
    .sort_index()
)


print("\n" + "=" * 80)
print("PER-WELL FEATURE IMPORTANCE")
print("=" * 80)

print(
    matrix.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# DOMINANT FEATURE BY WELL
# ============================================================

dominant_rows = []

for well, row in matrix.iterrows():

    feature_values = row.dropna()

    if feature_values.empty:
        continue

    dominant_feature = feature_values.idxmax()

    dominant_value = feature_values.max()

    dominant_rows.append(
        {
            "WELL": well,
            "DOMINANT_FEATURE": dominant_feature,
            "IMPORTANCE": dominant_value,
        }
    )


dominant_df = pd.DataFrame(
    dominant_rows
)


print("\n" + "=" * 80)
print("DOMINANT FEATURE BY HELD-OUT WELL")
print("=" * 80)

print(
    dominant_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(
    f"Unique wells analyzed : "
    f"{importance_df['HELD_OUT_WELL'].nunique()}"
)

print(
    f"Expected wells        : "
    f"{len(wells)}"
)

print(
    f"Features analyzed     : "
    f"{importance_df['FEATURE'].nunique()}"
)

print(
    f"Expected features     : "
    f"{len(FEATURES)}"
)

print("\nOutputs:")

print(IMPORTANCE_OUTPUT)
print(WELL_IMPORTANCE_OUTPUT)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)