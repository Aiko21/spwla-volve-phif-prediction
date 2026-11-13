from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

GENERALIZATION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "crosswell_failure_regimes.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "covariate_relationship_shift.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)
generalization = pd.read_csv(GENERALIZATION_FILE)

df.columns = df.columns.str.strip()
generalization.columns = generalization.columns.str.strip()


# ============================================================
# PREPARE FEATURES
# ============================================================

df["LOG10_RT"] = np.log10(
    df["RT"].clip(lower=1e-6)
)

df = df.dropna(
    subset=[
        "WELL",
        "GR",
        "LOG10_RT",
        "PHIF",
    ]
).copy()


# ============================================================
# HELPER
# ============================================================

def spearman_corr(data, x, y):
    """
    Return Spearman correlation.
    Requires at least two valid observations.
    """
    if len(data) < 2:
        return np.nan

    return data[x].corr(
        data[y],
        method="spearman",
    )


# ============================================================
# TRUE LEAVE-ONE-WELL-OUT DIAGNOSTIC
# ============================================================

rows = []

all_wells = sorted(
    df["WELL"].unique()
)

for held_out_well in all_wells:

    held_out = df[
        df["WELL"] == held_out_well
    ].copy()

    reference = df[
        df["WELL"] != held_out_well
    ].copy()

    # --------------------------------------------------------
    # Held-out statistics
    # --------------------------------------------------------

    gr_median = held_out["GR"].median()
    gr_iqr = (
        held_out["GR"].quantile(0.75)
        - held_out["GR"].quantile(0.25)
    )

    logrt_median = held_out["LOG10_RT"].median()
    logrt_iqr = (
        held_out["LOG10_RT"].quantile(0.75)
        - held_out["LOG10_RT"].quantile(0.25)
    )

    gr_phif_corr = spearman_corr(
        held_out,
        "GR",
        "PHIF",
    )

    logrt_phif_corr = spearman_corr(
        held_out,
        "LOG10_RT",
        "PHIF",
    )

    # --------------------------------------------------------
    # Leave-one-well-out reference statistics
    # --------------------------------------------------------

    ref_gr_median = reference["GR"].median()
    ref_gr_iqr = (
        reference["GR"].quantile(0.75)
        - reference["GR"].quantile(0.25)
    )

    ref_logrt_median = reference["LOG10_RT"].median()
    ref_logrt_iqr = (
        reference["LOG10_RT"].quantile(0.75)
        - reference["LOG10_RT"].quantile(0.25)
    )

    ref_gr_phif_corr = spearman_corr(
        reference,
        "GR",
        "PHIF",
    )

    ref_logrt_phif_corr = spearman_corr(
        reference,
        "LOG10_RT",
        "PHIF",
    )

    # --------------------------------------------------------
    # Distribution shifts
    # --------------------------------------------------------

    gr_median_shift = abs(
        gr_median - ref_gr_median
    )

    gr_iqr_shift = abs(
        gr_iqr - ref_gr_iqr
    )

    logrt_median_shift = abs(
        logrt_median - ref_logrt_median
    )

    logrt_iqr_shift = abs(
        logrt_iqr - ref_logrt_iqr
    )

    # --------------------------------------------------------
    # Relationship shifts
    # --------------------------------------------------------

    gr_relationship_shift = abs(
        gr_phif_corr - ref_gr_phif_corr
    )

    logrt_relationship_shift = abs(
        logrt_phif_corr - ref_logrt_phif_corr
    )

    rows.append(
        {
            "WELL": held_out_well,

            "GR_MEDIAN": gr_median,
            "GR_IQR": gr_iqr,
            "LOG10_RT_MEDIAN": logrt_median,
            "LOG10_RT_IQR": logrt_iqr,

            "REFERENCE_GR_MEDIAN": ref_gr_median,
            "REFERENCE_GR_IQR": ref_gr_iqr,
            "REFERENCE_LOG10_RT_MEDIAN": ref_logrt_median,
            "REFERENCE_LOG10_RT_IQR": ref_logrt_iqr,

            "GR_MEDIAN_SHIFT": gr_median_shift,
            "GR_IQR_SHIFT": gr_iqr_shift,
            "LOG10_RT_MEDIAN_SHIFT": logrt_median_shift,
            "LOG10_RT_IQR_SHIFT": logrt_iqr_shift,

            "GR_PHIF_SPEARMAN": gr_phif_corr,
            "REFERENCE_GR_PHIF_SPEARMAN": ref_gr_phif_corr,
            "GR_RELATIONSHIP_SHIFT": gr_relationship_shift,

            "LOG10_RT_PHIF_SPEARMAN": logrt_phif_corr,
            "REFERENCE_LOG10_RT_PHIF_SPEARMAN": (
                ref_logrt_phif_corr
            ),
            "LOG10_RT_RELATIONSHIP_SHIFT": (
                logrt_relationship_shift
            ),
        }
    )


shift = pd.DataFrame(rows)


# ============================================================
# ADD MODEL RESULTS
# ============================================================

generalization_cols = [
    "WELL",
    "XGB_R2",
    "LINEAR_R2",
    "XGB_GENERALIZATION_REGIME",
    "RT_PHIF_RELATIONSHIP",
    "RT_VALUE_REGIME",
]

shift = shift.merge(
    generalization[generalization_cols],
    on="WELL",
    how="left",
)


# ============================================================
# DESCRIPTIVE INDICES
# ============================================================

shift["COVARIATE_SHIFT_INDEX"] = (
    shift["GR_MEDIAN_SHIFT"]
    + shift["GR_IQR_SHIFT"]
    + shift["LOG10_RT_MEDIAN_SHIFT"]
    + shift["LOG10_RT_IQR_SHIFT"]
)

shift["RELATIONSHIP_SHIFT_INDEX"] = (
    shift["GR_RELATIONSHIP_SHIFT"]
    + shift["LOG10_RT_RELATIONSHIP_SHIFT"]
)

shift["TOTAL_DIAGNOSTIC_SHIFT"] = (
    shift["COVARIATE_SHIFT_INDEX"]
    + shift["RELATIONSHIP_SHIFT_INDEX"]
)


# ============================================================
# RANKS
# ============================================================

shift["COVARIATE_SHIFT_RANK"] = (
    shift["COVARIATE_SHIFT_INDEX"]
    .rank(
        ascending=False,
        method="min",
    )
)

shift["RELATIONSHIP_SHIFT_RANK"] = (
    shift["RELATIONSHIP_SHIFT_INDEX"]
    .rank(
        ascending=False,
        method="min",
    )
)

shift["TOTAL_SHIFT_RANK"] = (
    shift["TOTAL_DIAGNOSTIC_SHIFT"]
    .rank(
        ascending=False,
        method="min",
    )
)


# ============================================================
# SORT
# ============================================================

shift = shift.sort_values(
    "XGB_R2",
    ascending=False,
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

shift.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 80)
print("TRUE LEAVE-ONE-WELL-OUT SHIFT DIAGNOSTIC")
print("=" * 80)

print(
    "\nReference for each well:"
)

print(
    "All other wells except the held-out well."
)

print(
    "\nOutput:"
)

print(OUTPUT_FILE)


display_cols = [
    "WELL",
    "XGB_R2",
    "XGB_GENERALIZATION_REGIME",
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
]

print("\n" + "=" * 80)
print("WELL-LEVEL SHIFT DIAGNOSTIC")
print("=" * 80)

print(
    shift[
        display_cols
    ].to_string(index=False)
)


# ============================================================
# REGIME SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SHIFT BY GENERALIZATION REGIME")
print("=" * 80)

for regime in [
    "GOOD",
    "MODERATE",
    "POOR",
]:

    subset = shift[
        shift["XGB_GENERALIZATION_REGIME"]
        == regime
    ]

    if subset.empty:
        continue

    print(f"\n{regime} ({len(subset)} wells):")

    print(
        f"  Covariate shift mean   = "
        f"{subset['COVARIATE_SHIFT_INDEX'].mean():.6f}"
    )

    print(
        f"  Relationship shift mean = "
        f"{subset['RELATIONSHIP_SHIFT_INDEX'].mean():.6f}"
    )

    print(
        f"  Total shift mean       = "
        f"{subset['TOTAL_DIAGNOSTIC_SHIFT'].mean():.6f}"
    )


# ============================================================
# EXPLORATORY CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATORY SPEARMAN CORRELATION WITH XGB R2")
print("=" * 80)

corr_cols = [
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
]

corr_rows = []

for col in corr_cols:

    corr = shift[col].corr(
        shift["XGB_R2"],
        method="spearman",
    )

    corr_rows.append(
        {
            "VARIABLE": col,
            "SPEARMAN_WITH_XGB_R2": corr,
        }
    )

corr_table = pd.DataFrame(
    corr_rows
)

print(
    corr_table.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)