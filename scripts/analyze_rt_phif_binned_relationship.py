from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "rt_phif_binned_relationship.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_cols = [
    "WELL",
    "PHIF",
    "RT",
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}"
    )


# ============================================================
# PREPARE DATA
# ============================================================

df = df[required_cols].copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

df = df.dropna(
    subset=["WELL", "PHIF", "RT"]
)

# RT must be positive before log transform.
df = df[df["RT"] > 0].copy()

df["LOG10_RT"] = np.log10(df["RT"])


# ============================================================
# BIN WITHIN EACH WELL
# ============================================================
#
# qcut is performed separately for every well.
#
# Therefore:
# Q1 = lowest 20% RT values within that well
# Q5 = highest 20% RT values within that well
#
# This avoids confusing between-well RT distribution
# differences with within-well RT -> PHIF behavior.
# ============================================================

results = []

for well, group in df.groupby("WELL", sort=True):

    group = group.copy()

    try:
        group["RT_BIN"] = pd.qcut(
            group["LOG10_RT"],
            q=5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
            duplicates="drop",
        )
    except ValueError:
        print(
            f"Skipping {well}: insufficient unique RT values."
        )
        continue

    grouped = (
        group
        .groupby(
            "RT_BIN",
            observed=False,
        )
        .agg(
            N=("PHIF", "size"),
            RT_MEAN=("RT", "mean"),
            RT_MEDIAN=("RT", "median"),
            LOG10_RT_MEAN=("LOG10_RT", "mean"),
            LOG10_RT_MEDIAN=("LOG10_RT", "median"),
            PHIF_MEAN=("PHIF", "mean"),
            PHIF_MEDIAN=("PHIF", "median"),
            PHIF_STD=("PHIF", "std"),
        )
        .reset_index()
    )

    grouped["WELL"] = well

    results.append(grouped)


# ============================================================
# COMBINE
# ============================================================

binned = pd.concat(
    results,
    ignore_index=True,
)


# ============================================================
# ORDER COLUMNS
# ============================================================

binned = binned[
    [
        "WELL",
        "RT_BIN",
        "N",
        "RT_MEAN",
        "RT_MEDIAN",
        "LOG10_RT_MEAN",
        "LOG10_RT_MEDIAN",
        "PHIF_MEAN",
        "PHIF_MEDIAN",
        "PHIF_STD",
    ]
]


# ============================================================
# CALCULATE Q1 -> Q5 CHANGE
# ============================================================

summary = []

for well, group in binned.groupby("WELL", sort=True):

    group = group.sort_values("RT_BIN")

    if len(group) < 5:
        continue

    q1 = group.iloc[0]
    q5 = group.iloc[-1]

    delta_mean = (
        q5["PHIF_MEAN"]
        - q1["PHIF_MEAN"]
    )

    delta_median = (
        q5["PHIF_MEDIAN"]
        - q1["PHIF_MEDIAN"]
    )

    summary.append(
        {
            "WELL": well,
            "Q1_PHIF_MEAN": q1["PHIF_MEAN"],
            "Q5_PHIF_MEAN": q5["PHIF_MEAN"],
            "DELTA_PHIF_MEAN_Q1_Q5": delta_mean,
            "Q1_PHIF_MEDIAN": q1["PHIF_MEDIAN"],
            "Q5_PHIF_MEDIAN": q5["PHIF_MEDIAN"],
            "DELTA_PHIF_MEDIAN_Q1_Q5": delta_median,
        }
    )


summary_df = pd.DataFrame(summary)


# ============================================================
# SIMPLE MONOTONICITY DIAGNOSTIC
# ============================================================
#
# We calculate Spearman correlation between bin order
# (1..5) and PHIF mean / median.
#
# This is NOT the same as the original row-level
# Spearman correlation.
#
# It measures whether the binned PHIF response tends
# to increase/decrease monotonically across RT bins.
# ============================================================

monotonicity = []

for well, group in binned.groupby("WELL", sort=True):

    group = group.sort_values("RT_BIN").copy()

    group["BIN_ORDER"] = range(1, len(group) + 1)

    mean_corr = group[
        ["BIN_ORDER", "PHIF_MEAN"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    median_corr = group[
        ["BIN_ORDER", "PHIF_MEDIAN"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    monotonicity.append(
        {
            "WELL": well,
            "BINNED_PHIF_MEAN_SPEARMAN": mean_corr,
            "BINNED_PHIF_MEDIAN_SPEARMAN": median_corr,
        }
    )


monotonicity_df = pd.DataFrame(monotonicity)


summary_df = summary_df.merge(
    monotonicity_df,
    on="WELL",
    how="left",
)


# ============================================================
# PRINT DETAILED BIN TABLE
# ============================================================

print("=" * 110)
print("RT -> PHIF BINNED RELATIONSHIP BY WELL")
print("=" * 110)
print()

print(
    binned[
        [
            "WELL",
            "RT_BIN",
            "N",
            "RT_MEDIAN",
            "LOG10_RT_MEDIAN",
            "PHIF_MEAN",
            "PHIF_MEDIAN",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("Q1 -> Q5 RT EFFECT SUMMARY")
print("=" * 110)
print()

print(
    summary_df[
        [
            "WELL",
            "Q1_PHIF_MEAN",
            "Q5_PHIF_MEAN",
            "DELTA_PHIF_MEAN_Q1_Q5",
            "Q1_PHIF_MEDIAN",
            "Q5_PHIF_MEDIAN",
            "DELTA_PHIF_MEDIAN_Q1_Q5",
            "BINNED_PHIF_MEAN_SPEARMAN",
            "BINNED_PHIF_MEDIAN_SPEARMAN",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

binned.to_csv(
    OUTPUT_PATH,
    index=False,
)

SUMMARY_OUTPUT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "rt_phif_binned_summary.csv"
)

summary_df.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False,
)


print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)
print(OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)
print()
print("Done.")