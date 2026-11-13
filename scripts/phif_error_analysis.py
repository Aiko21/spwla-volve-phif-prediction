from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "reports"
    / "models"
    / "phif_final_test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["LOG10_RT"] = np.log10(df["RT"])

df["RESIDUAL"] = (
    df["PHIF"]
    - df["PHIF_PRED"]
)

df["ABS_ERROR"] = (
    df["RESIDUAL"].abs()
)


# ============================================================
# OVERALL ERROR
# ============================================================

print("=" * 70)
print("PHIF ERROR ANALYSIS V1")
print("=" * 70)

print(f"\nRows: {len(df):,}")


print("\n" + "=" * 70)
print("OVERALL ERROR SUMMARY")
print("=" * 70)

print(
    df[
        [
            "PHIF",
            "PHIF_PRED",
            "RESIDUAL",
            "ABS_ERROR",
        ]
    ]
    .describe()
    .round(6)
    .to_string()
)


# ============================================================
# ERROR BY WELL
# ============================================================

well_error = (
    df.groupby("WELL")
    .agg(
        ROWS=("PHIF", "size"),

        ACTUAL_MEAN=("PHIF", "mean"),
        PRED_MEAN=("PHIF_PRED", "mean"),

        ACTUAL_STD=("PHIF", "std"),
        PRED_STD=("PHIF_PRED", "std"),

        MEAN_RESIDUAL=("RESIDUAL", "mean"),
        MAE=("ABS_ERROR", "mean"),

        MAX_ABS_ERROR=("ABS_ERROR", "max"),
    )
)

print("\n" + "=" * 70)
print("ERROR BY WELL")
print("=" * 70)

print(
    well_error
    .round(6)
    .to_string()
)


# ============================================================
# RESIDUAL CORRELATIONS
# ============================================================

numeric_cols = [
    "DEPTH",
    "GR",
    "RT",
    "LOG10_RT",
    "PHIF",
    "PHIF_PRED",
    "RESIDUAL",
    "ABS_ERROR",
]

pearson_corr = (
    df[numeric_cols]
    .corr(method="pearson")
)

spearman_corr = (
    df[numeric_cols]
    .corr(method="spearman")
)

print("\n" + "=" * 70)
print("PEARSON CORRELATION WITH RESIDUAL")
print("=" * 70)

print(
    pearson_corr["RESIDUAL"]
    .sort_values()
    .round(4)
    .to_string()
)

print("\n" + "=" * 70)
print("SPEARMAN CORRELATION WITH RESIDUAL")
print("=" * 70)

print(
    spearman_corr["RESIDUAL"]
    .sort_values()
    .round(4)
    .to_string()
)


# ============================================================
# ERROR BY PHIF RANGE
# ============================================================

df["PHIF_BIN"] = pd.cut(
    df["PHIF"],
    bins=[
        -np.inf,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        np.inf,
    ],
)

error_by_phif = (
    df.groupby(
        "PHIF_BIN",
        observed=True
    )
    .agg(
        ROWS=("PHIF", "size"),
        MEAN_ACTUAL=("PHIF", "mean"),
        MEAN_PRED=("PHIF_PRED", "mean"),
        MEAN_RESIDUAL=("RESIDUAL", "mean"),
        MAE=("ABS_ERROR", "mean"),
    )
)

print("\n" + "=" * 70)
print("ERROR BY ACTUAL PHIF RANGE")
print("=" * 70)

print(
    error_by_phif
    .round(6)
    .to_string()
)


# ============================================================
# ERROR BY DEPTH QUANTILE
# ============================================================

df["DEPTH_BIN"] = pd.qcut(
    df["DEPTH"],
    q=10,
    duplicates="drop",
)

error_by_depth = (
    df.groupby(
        ["WELL", "DEPTH_BIN"],
        observed=True
    )
    .agg(
        ROWS=("PHIF", "size"),
        MEAN_ACTUAL=("PHIF", "mean"),
        MEAN_PRED=("PHIF_PRED", "mean"),
        MEAN_RESIDUAL=("RESIDUAL", "mean"),
        MAE=("ABS_ERROR", "mean"),
    )
    .reset_index()
)

print("\n" + "=" * 70)
print("ERROR BY WELL / DEPTH DECILE")
print("=" * 70)

print(
    error_by_depth
    .round(6)
    .to_string(index=False)
)


# ============================================================
# WORST PREDICTIONS
# ============================================================

worst = (
    df[
        [
            "WELL",
            "DEPTH",
            "GR",
            "RT",
            "LOG10_RT",
            "PHIF",
            "PHIF_PRED",
            "RESIDUAL",
            "ABS_ERROR",
        ]
    ]
    .sort_values(
        "ABS_ERROR",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("TOP 30 WORST PREDICTIONS")
print("=" * 70)

print(
    worst
    .head(30)
    .round(6)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

well_error.to_csv(
    OUTPUT_DIR
    / "phif_error_by_well.csv"
)

error_by_phif.to_csv(
    OUTPUT_DIR
    / "phif_error_by_target_range.csv"
)

error_by_depth.to_csv(
    OUTPUT_DIR
    / "phif_error_by_depth_decile.csv"
)

worst.to_csv(
    OUTPUT_DIR
    / "phif_worst_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("ERROR ANALYSIS FILES")
print("=" * 70)

print(
    OUTPUT_DIR
    / "phif_error_by_well.csv"
)

print(
    OUTPUT_DIR
    / "phif_error_by_target_range.csv"
)

print(
    OUTPUT_DIR
    / "phif_error_by_depth_decile.csv"
)

print(
    OUTPUT_DIR
    / "phif_worst_predictions.csv"
)

print("\nDONE.")
