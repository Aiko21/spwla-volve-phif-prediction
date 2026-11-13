from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phif_train.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "data_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(TRAIN_FILE)

print("=" * 70)
print("PHIF FEATURE-TARGET EDA")
print("=" * 70)

print(f"\nRows : {len(df):,}")
print(f"Wells: {df['WELL'].nunique()}")


# ============================================================
# NUMERIC SUMMARY
# ============================================================

columns = [
    "GR",
    "RT",
    "PHIF",
]

summary = (
    df[columns]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )
    .T
)

print("\n" + "=" * 70)
print("NUMERIC SUMMARY")
print("=" * 70)

print(
    summary.round(4).to_string()
)


# ============================================================
# SKEWNESS
# ============================================================

skewness = (
    df[columns]
    .skew()
    .rename("skewness")
)

print("\n" + "=" * 70)
print("SKEWNESS")
print("=" * 70)

print(
    skewness.round(4).to_string()
)


# ============================================================
# CORRELATION
# ============================================================

correlation = (
    df[columns]
    .corr(method="pearson")
)

spearman = (
    df[columns]
    .corr(method="spearman")
)

print("\n" + "=" * 70)
print("PEARSON CORRELATION")
print("=" * 70)

print(
    correlation.round(4).to_string()
)

print("\n" + "=" * 70)
print("SPEARMAN CORRELATION")
print("=" * 70)

print(
    spearman.round(4).to_string()
)


# ============================================================
# PER-WELL CORRELATION
# ============================================================

well_corr = []

for well, group in df.groupby("WELL"):

    pearson_gr = group["GR"].corr(
        group["PHIF"],
        method="pearson"
    )

    spearman_gr = group["GR"].corr(
        group["PHIF"],
        method="spearman"
    )

    pearson_rt = group["RT"].corr(
        group["PHIF"],
        method="pearson"
    )

    spearman_rt = group["RT"].corr(
        group["PHIF"],
        method="spearman"
    )

    well_corr.append(
        {
            "WELL": well,
            "ROWS": len(group),
            "PEARSON_GR_PHIF": pearson_gr,
            "SPEARMAN_GR_PHIF": spearman_gr,
            "PEARSON_RT_PHIF": pearson_rt,
            "SPEARMAN_RT_PHIF": spearman_rt,
        }
    )


well_corr_df = pd.DataFrame(
    well_corr
)

print("\n" + "=" * 70)
print("PER-WELL CORRELATION")
print("=" * 70)

print(
    well_corr_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_DIR
    / "phif_feature_summary_train.csv"
)

correlation.to_csv(
    OUTPUT_DIR
    / "phif_correlation_pearson_train.csv"
)

spearman.to_csv(
    OUTPUT_DIR
    / "phif_correlation_spearman_train.csv"
)

well_corr_df.to_csv(
    OUTPUT_DIR
    / "phif_per_well_correlation_train.csv",
    index=False
)


print("\n" + "=" * 70)
print("EDA STATISTICS SAVED")
print("=" * 70)

print(
    OUTPUT_DIR
    / "phif_feature_summary_train.csv"
)

print(
    OUTPUT_DIR
    / "phif_per_well_correlation_train.csv"
)
