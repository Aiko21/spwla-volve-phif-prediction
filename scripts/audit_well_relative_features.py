from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "well_relative_feature_audit.csv"
)


print("=" * 78)
print("WELL-RELATIVE FEATURE AUDIT")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

print()
print(f"Dataset shape: {df.shape}")

# ------------------------------------------------------------------
# Derived feature
# ------------------------------------------------------------------

df["LOG10_RT"] = np.log10(df["RT"])

# ------------------------------------------------------------------
# Well-relative standardization
#
# IMPORTANT:
# This is a descriptive diagnostic only.
# These values must NOT be used directly in the final CV model
# unless the well-level parameters are handled without leakage.
# ------------------------------------------------------------------

df["GR_ZWELL"] = (
    df["GR"] - df.groupby("WELL")["GR"].transform("mean")
) / df.groupby("WELL")["GR"].transform("std")

df["LOGRT_ZWELL"] = (
    df["LOG10_RT"]
    - df.groupby("WELL")["LOG10_RT"].transform("mean")
) / df.groupby("WELL")["LOG10_RT"].transform("std")


# ------------------------------------------------------------------
# Summary before / after
# ------------------------------------------------------------------

rows = []

for well, group in df.groupby("WELL"):

    rows.append(
        {
            "WELL": well,
            "N_ROWS": len(group),

            "GR_MEAN": group["GR"].mean(),
            "GR_STD": group["GR"].std(),
            "GR_ZWELL_MEAN": group["GR_ZWELL"].mean(),
            "GR_ZWELL_STD": group["GR_ZWELL"].std(),

            "LOGRT_MEAN": group["LOG10_RT"].mean(),
            "LOGRT_STD": group["LOG10_RT"].std(),
            "LOGRT_ZWELL_MEAN": group["LOGRT_ZWELL"].mean(),
            "LOGRT_ZWELL_STD": group["LOGRT_ZWELL"].std(),

            "GR_ZWELL_MIN": group["GR_ZWELL"].min(),
            "GR_ZWELL_P01": group["GR_ZWELL"].quantile(0.01),
            "GR_ZWELL_P05": group["GR_ZWELL"].quantile(0.05),
            "GR_ZWELL_MEDIAN": group["GR_ZWELL"].median(),
            "GR_ZWELL_P95": group["GR_ZWELL"].quantile(0.95),
            "GR_ZWELL_P99": group["GR_ZWELL"].quantile(0.99),
            "GR_ZWELL_MAX": group["GR_ZWELL"].max(),

            "LOGRT_ZWELL_MIN": group["LOGRT_ZWELL"].min(),
            "LOGRT_ZWELL_P01": group["LOGRT_ZWELL"].quantile(0.01),
            "LOGRT_ZWELL_P05": group["LOGRT_ZWELL"].quantile(0.05),
            "LOGRT_ZWELL_MEDIAN": group["LOGRT_ZWELL"].median(),
            "LOGRT_ZWELL_P95": group["LOGRT_ZWELL"].quantile(0.95),
            "LOGRT_ZWELL_P99": group["LOGRT_ZWELL"].quantile(0.99),
            "LOGRT_ZWELL_MAX": group["LOGRT_ZWELL"].max(),
        }
    )

summary = pd.DataFrame(rows)


# ------------------------------------------------------------------
# Print diagnostics
# ------------------------------------------------------------------

print()
print("=" * 78)
print("WELL-RELATIVE SUMMARY")
print("=" * 78)

print(
    summary[
        [
            "WELL",
            "N_ROWS",
            "GR_MEAN",
            "GR_STD",
            "GR_ZWELL_MEAN",
            "GR_ZWELL_STD",
            "LOGRT_MEAN",
            "LOGRT_STD",
            "LOGRT_ZWELL_MEAN",
            "LOGRT_ZWELL_STD",
        ]
    ].to_string(index=False)
)


print()
print("=" * 78)
print("RANGE COMPARISON")
print("=" * 78)

for feature in ["GR", "LOG10_RT", "GR_ZWELL", "LOGRT_ZWELL"]:

    stats = df[feature].agg(
        ["min", "mean", "std", "median", "max"]
    )

    print()
    print(feature)
    print(stats.to_string())


# ------------------------------------------------------------------
# Cross-well variability of summary statistics
# ------------------------------------------------------------------

print()
print("=" * 78)
print("CROSS-WELL VARIABILITY OF FEATURE MEANS")
print("=" * 78)

for feature in [
    "GR_MEAN",
    "GR_ZWELL_MEAN",
    "LOGRT_MEAN",
    "LOGRT_ZWELL_MEAN",
]:

    values = summary[feature]

    print(
        f"{feature:25s}: "
        f"min={values.min():+.6f} "
        f"max={values.max():+.6f} "
        f"std={values.std():.6f}"
    )


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

summary.to_csv(OUTPUT_PATH, index=False)

print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)
print(OUTPUT_PATH)

print()
print("Done.")