from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

SPLIT_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "phif_well_split_metadata.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "data_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FEATURE DISTRIBUTION SHIFT AUDIT")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
split_df = pd.read_csv(SPLIT_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Split metadata shape: {split_df.shape}")


# ============================================================
# NORMALIZE SPLIT METADATA
# ============================================================

split_df["SPLIT"] = (
    split_df["SPLIT"]
    .astype(str)
    .str.upper()
    .str.strip()
)

split_df["WELL"] = (
    split_df["WELL"]
    .astype(str)
    .str.strip()
)


# ============================================================
# MERGE SPLIT LABEL
# ============================================================

df = df.merge(
    split_df[
        [
            "WELL",
            "SPLIT",
        ]
    ],
    on="WELL",
    how="left",
    validate="many_to_one",
)


if df["SPLIT"].isna().any():
    missing_wells = (
        df.loc[
            df["SPLIT"].isna(),
            "WELL",
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        f"Missing split assignment for wells: {missing_wells}"
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["LOG10_RT"] = np.log10(
    df["RT"].clip(lower=1e-6)
)


FEATURES = [
    "GR",
    "RT",
    "LOG10_RT",
]


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def calculate_statistics(group, feature):

    values = group[feature].dropna()

    if len(values) == 0:
        return {
            "COUNT": 0,
            "MEAN": np.nan,
            "STD": np.nan,
            "MIN": np.nan,
            "P01": np.nan,
            "P05": np.nan,
            "P25": np.nan,
            "MEDIAN": np.nan,
            "P75": np.nan,
            "P95": np.nan,
            "P99": np.nan,
            "MAX": np.nan,
        }

    return {
        "COUNT": len(values),
        "MEAN": values.mean(),
        "STD": values.std(),
        "MIN": values.min(),
        "P01": values.quantile(0.01),
        "P05": values.quantile(0.05),
        "P25": values.quantile(0.25),
        "MEDIAN": values.median(),
        "P75": values.quantile(0.75),
        "P95": values.quantile(0.95),
        "P99": values.quantile(0.99),
        "MAX": values.max(),
    }


# ============================================================
# SPLIT-LEVEL STATISTICS
# ============================================================

records = []

for split_name, group in df.groupby("SPLIT"):

    for feature in FEATURES:

        stats = calculate_statistics(
            group,
            feature,
        )

        records.append(
            {
                "SPLIT": split_name,
                "FEATURE": feature,
                **stats,
            }
        )


split_stats_df = pd.DataFrame(records)


# ============================================================
# SAVE SPLIT STATISTICS
# ============================================================

split_stats_path = (
    OUTPUT_DIR /
    "feature_distribution_by_split.csv"
)

split_stats_df.to_csv(
    split_stats_path,
    index=False,
)


# ============================================================
# PRINT SPLIT STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION BY SPLIT")
print("=" * 80)

for feature in FEATURES:

    print(f"\n--- {feature} ---")

    display_df = split_stats_df[
        split_stats_df["FEATURE"] == feature
    ][
        [
            "SPLIT",
            "COUNT",
            "MEAN",
            "STD",
            "MIN",
            "P01",
            "P05",
            "MEDIAN",
            "P75",
            "P95",
            "P99",
            "MAX",
        ]
    ]

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# ============================================================
# PER-WELL STATISTICS
# ============================================================

well_records = []

for well, group in df.groupby("WELL"):

    split_name = group["SPLIT"].iloc[0]

    for feature in FEATURES:

        stats = calculate_statistics(
            group,
            feature,
        )

        well_records.append(
            {
                "WELL": well,
                "SPLIT": split_name,
                "FEATURE": feature,
                **stats,
            }
        )


well_stats_df = pd.DataFrame(
    well_records
)


# ============================================================
# SAVE WELL STATISTICS
# ============================================================

well_stats_path = (
    OUTPUT_DIR /
    "feature_distribution_by_well.csv"
)

well_stats_df.to_csv(
    well_stats_path,
    index=False,
)


# ============================================================
# PRINT PER-WELL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PER-WELL FEATURE DISTRIBUTION")
print("=" * 80)

for feature in FEATURES:

    print(f"\n--- {feature} ---")

    display_df = well_stats_df[
        well_stats_df["FEATURE"] == feature
    ][
        [
            "WELL",
            "SPLIT",
            "COUNT",
            "MEAN",
            "STD",
            "MIN",
            "P05",
            "MEDIAN",
            "P95",
            "P99",
            "MAX",
        ]
    ]

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# ============================================================
# TEST WELLS VS TRAIN RANGE
# ============================================================

print("\n" + "=" * 80)
print("TEST WELL VALUES OUTSIDE TRAIN FEATURE RANGE")
print("=" * 80)

train_df = df[
    df["SPLIT"] == "TRAIN"
]

test_df = df[
    df["SPLIT"] == "TEST"
]

range_records = []

for feature in FEATURES:

    train_values = train_df[feature].dropna()

    train_min = train_values.min()
    train_max = train_values.max()

    for well, group in test_df.groupby("WELL"):

        values = group[feature].dropna()

        below = (values < train_min).sum()
        above = (values > train_max).sum()

        total = len(values)

        range_records.append(
            {
                "WELL": well,
                "FEATURE": feature,
                "TRAIN_MIN": train_min,
                "TRAIN_MAX": train_max,
                "TEST_COUNT": total,
                "BELOW_TRAIN_MIN": below,
                "ABOVE_TRAIN_MAX": above,
                "OUTSIDE_COUNT": below + above,
                "OUTSIDE_PCT": (
                    (below + above) / total * 100
                    if total > 0
                    else np.nan
                ),
            }
        )


range_df = pd.DataFrame(
    range_records
)


# ============================================================
# SAVE RANGE COMPARISON
# ============================================================

range_path = (
    OUTPUT_DIR /
    "test_vs_train_feature_range.csv"
)

range_df.to_csv(
    range_path,
    index=False,
)


print(
    range_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print("\nSaved:")
print(split_stats_path)
print(well_stats_path)
print(range_path)
