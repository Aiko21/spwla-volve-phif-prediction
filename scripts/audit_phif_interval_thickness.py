from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERVAL_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_high_quality_intervals.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_interval_thickness_audit.csv"
)


# ============================================================
# LOAD
# ============================================================

if not INTERVAL_FILE.exists():
    raise FileNotFoundError(
        f"Interval file not found: {INTERVAL_FILE}"
    )

df = pd.read_csv(INTERVAL_FILE)

required_cols = [
    "WELL",
    "TOP_DEPTH",
    "BASE_DEPTH",
    "THICKNESS",
    "N_HIGH_SAMPLES",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "INTERVAL_MAE",
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DEPTH / THICKNESS VALIDATION
# ============================================================

df["THICKNESS_CALC"] = (
    df["BASE_DEPTH"] - df["TOP_DEPTH"]
)

df["THICKNESS_DIFF"] = (
    df["THICKNESS"] - df["THICKNESS_CALC"]
)

df["SAMPLE_STEP_APPROX"] = np.where(
    df["N_HIGH_SAMPLES"] > 1,
    df["THICKNESS"]
    / (df["N_HIGH_SAMPLES"] - 1),
    np.nan,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

thickness_quantiles = (
    df["THICKNESS"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
)

sample_quantiles = (
    df["N_HIGH_SAMPLES"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
)

step_quantiles = (
    df["SAMPLE_STEP_APPROX"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
)


# ============================================================
# THICKNESS BINS
# ============================================================

bins = [
    -np.inf,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    np.inf,
]

labels = [
    "<0.5",
    "0.5-<1",
    "1-<2",
    "2-<5",
    "5-<10",
    "10-<20",
    ">=20",
]

df["THICKNESS_BIN"] = pd.cut(
    df["THICKNESS"],
    bins=bins,
    labels=labels,
    right=False,
)

thickness_distribution = (
    df["THICKNESS_BIN"]
    .value_counts(sort=False)
    .rename_axis("THICKNESS_BIN")
    .reset_index(name="N_INTERVALS")
)

thickness_distribution["PCT_INTERVALS"] = (
    thickness_distribution["N_INTERVALS"]
    / len(df)
    * 100
)


# ============================================================
# SUMMARY BY WELL
# ============================================================

summary = (
    df.groupby("WELL", observed=True)
    .agg(
        N_INTERVALS=("WELL", "size"),
        TOTAL_THICKNESS=("THICKNESS", "sum"),
        MEDIAN_THICKNESS=("THICKNESS", "median"),
        MEAN_THICKNESS=("THICKNESS", "mean"),
        MIN_THICKNESS=("THICKNESS", "min"),
        MAX_THICKNESS=("THICKNESS", "max"),
        MEDIAN_HIGH_SAMPLES=("N_HIGH_SAMPLES", "median"),
        MIN_HIGH_SAMPLES=("N_HIGH_SAMPLES", "min"),
        MAX_HIGH_SAMPLES=("N_HIGH_SAMPLES", "max"),
        MEDIAN_STEP_APPROX=("SAMPLE_STEP_APPROX", "median"),
        MEAN_INTERVAL_MAE=("INTERVAL_MAE", "mean"),
        MEDIAN_INTERVAL_MAE=("INTERVAL_MAE", "median"),
    )
    .reset_index()
)


# ============================================================
# VALIDATION
# ============================================================

max_thickness_diff = (
    df["THICKNESS_DIFF"].abs().max()
)


# ============================================================
# PRINT
# ============================================================

print("=" * 80)
print("PHIF HIGH-QUALITY INTERVAL THICKNESS AUDIT")
print("=" * 80)

print("\nInput file:")
print(INTERVAL_FILE)

print(f"\nTotal intervals: {len(df):,}")
print(f"Total wells: {df['WELL'].nunique():,}")

print("\n--- THICKNESS QUANTILES ---")
print(thickness_quantiles.to_string())

print("\n--- NUMBER OF HIGH-PHIF SAMPLES QUANTILES ---")
print(sample_quantiles.to_string())

print("\n--- APPROXIMATE SAMPLE STEP QUANTILES ---")
print(step_quantiles.to_string())

print("\n--- THICKNESS DISTRIBUTION ---")
print(
    thickness_distribution.to_string(index=False)
)

print("\n--- SUMMARY BY WELL ---")
print(
    summary.to_string(index=False)
)

print("\n--- VALIDATION ---")

print(
    "Maximum thickness calculation difference:",
    f"{max_thickness_diff:.10f}",
)

print(
    "\nIntervals with thickness < 0.5:",
    (df["THICKNESS"] < 0.5).sum(),
)

print(
    "Intervals with thickness >= 0.5:",
    (df["THICKNESS"] >= 0.5).sum(),
)

print(
    "Intervals with thickness >= 1.0:",
    (df["THICKNESS"] >= 1.0).sum(),
)

print(
    "Intervals with thickness >= 2.0:",
    (df["THICKNESS"] >= 2.0).sum(),
)

print(
    "Intervals with thickness >= 5.0:",
    (df["THICKNESS"] >= 5.0).sum(),
)

print("\n--- LONGEST INTERVALS ---")

print(
    df[
        [
            "WELL",
            "TOP_DEPTH",
            "BASE_DEPTH",
            "THICKNESS",
            "N_HIGH_SAMPLES",
            "HIGH_SAMPLE_PCT",
            "PREDICTED_PHIF_MEAN",
            "ACTUAL_PHIF_MEAN",
            "INTERVAL_MAE",
        ]
    ]
    .sort_values(
        "THICKNESS",
        ascending=False,
    )
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved summary:")
print(OUTPUT_FILE)