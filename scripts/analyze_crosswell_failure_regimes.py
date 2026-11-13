from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports" / "data_quality"
MODEL_DIR = PROJECT_ROOT / "reports" / "models"

WELL_GENERALIZATION_FILE = (
    REPORT_DIR / "well_generalization_diagnostic.csv"
)

RT_INCREMENTAL_FILE = (
    REPORT_DIR / "rt_incremental_value_by_well.csv"
)

INTERACTION_FILE = (
    MODEL_DIR / "phif_interaction_cv_results.csv"
)

OUTPUT_FILE = (
    REPORT_DIR / "crosswell_failure_regimes.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

well = pd.read_csv(WELL_GENERALIZATION_FILE)
rt = pd.read_csv(RT_INCREMENTAL_FILE)
interaction = pd.read_csv(INTERACTION_FILE)


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

well.columns = well.columns.str.strip()
rt.columns = rt.columns.str.strip()
interaction.columns = interaction.columns.str.strip()


# ============================================================
# SELECT WELL-LEVEL CHARACTERISTICS
# ============================================================

well_cols = [
    "WELL",
    "PHIF_MEDIAN",
    "GR_MEDIAN",
    "RT_MEDIAN",
    "LOG10_RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LINEAR_R2",
    "XGB_R2",
]

well_base = well[well_cols].copy()


# ============================================================
# RT INCREMENTAL VALUE
# ============================================================

# Keep XGBoost as the primary reference model for the
# incremental RT diagnostic.
rt_xgb = rt[
    rt["MODEL"].str.upper().eq("XGBOOST")
].copy()

rt_xgb = rt_xgb[
    [
        "HELD_OUT_WELL",
        "DELTA_MAE",
        "DELTA_RMSE",
        "DELTA_R2",
    ]
].copy()

rt_xgb = rt_xgb.rename(
    columns={
        "HELD_OUT_WELL": "WELL",
        "DELTA_MAE": "RT_DELTA_MAE_XGB",
        "DELTA_RMSE": "RT_DELTA_RMSE_XGB",
        "DELTA_R2": "RT_DELTA_R2_XGB",
    }
)


# ============================================================
# INTERACTION INCREMENTAL VALUE
# ============================================================

# The interaction CV file stores absolute metrics for both:
# BASE_GR_LOGRT and INTERACTION_GR_LOGRT.
#
# Therefore, calculate the incremental effect explicitly:
#
# Delta MAE  = Interaction MAE  - Base MAE
# Delta RMSE = Interaction RMSE - Base RMSE
# Delta R2   = Interaction R2   - Base R2
#
# For R2:
#   positive delta -> interaction improves performance
#   negative delta -> interaction hurts performance

interaction_xgb = interaction[
    interaction["MODEL"].str.upper().eq("XGBOOST")
].copy()

interaction_base = interaction_xgb[
    interaction_xgb["FEATURE_SET"].eq("BASE_GR_LOGRT")
][
    [
        "HELD_OUT_WELL",
        "MAE",
        "RMSE",
        "R2",
    ]
].copy()

interaction_enhanced = interaction_xgb[
    interaction_xgb["FEATURE_SET"].eq("INTERACTION_GR_LOGRT")
][
    [
        "HELD_OUT_WELL",
        "MAE",
        "RMSE",
        "R2",
    ]
].copy()

interaction_base = interaction_base.rename(
    columns={
        "MAE": "BASE_MAE",
        "RMSE": "BASE_RMSE",
        "R2": "BASE_R2",
    }
)

interaction_enhanced = interaction_enhanced.rename(
    columns={
        "MAE": "INTERACTION_MAE",
        "RMSE": "INTERACTION_RMSE",
        "R2": "INTERACTION_R2",
    }
)

interaction_xgb = interaction_base.merge(
    interaction_enhanced,
    on="HELD_OUT_WELL",
    how="inner",
)

interaction_xgb["DELTA_MAE"] = (
    interaction_xgb["INTERACTION_MAE"]
    - interaction_xgb["BASE_MAE"]
)

interaction_xgb["DELTA_RMSE"] = (
    interaction_xgb["INTERACTION_RMSE"]
    - interaction_xgb["BASE_RMSE"]
)

interaction_xgb["DELTA_R2"] = (
    interaction_xgb["INTERACTION_R2"]
    - interaction_xgb["BASE_R2"]
)

interaction_xgb = interaction_xgb[
    [
        "HELD_OUT_WELL",
        "DELTA_MAE",
        "DELTA_RMSE",
        "DELTA_R2",
    ]
].copy()

interaction_xgb = interaction_xgb.rename(
    columns={
        "HELD_OUT_WELL": "WELL",
        "DELTA_MAE": "INTERACTION_DELTA_MAE_XGB",
        "DELTA_RMSE": "INTERACTION_DELTA_RMSE_XGB",
        "DELTA_R2": "INTERACTION_DELTA_R2_XGB",
    }
)

# ============================================================
# MERGE
# ============================================================

result = well_base.merge(
    rt_xgb,
    on="WELL",
    how="left",
)

result = result.merge(
    interaction_xgb,
    on="WELL",
    how="left",
)


# ============================================================
# GENERALIZATION CLASSIFICATION
# ============================================================

# This is intentionally DESCRIPTIVE.
# It is not an ML target and not a statistical significance test.

def classify_generalization(r2):
    if pd.isna(r2):
        return "UNKNOWN"

    if r2 >= 0.20:
        return "GOOD"

    if r2 >= 0.00:
        return "MODERATE"

    return "POOR"


result["XGB_GENERALIZATION_REGIME"] = (
    result["XGB_R2"]
    .apply(classify_generalization)
)


# ============================================================
# RT VALUE CLASSIFICATION
# ============================================================

def classify_rt_value(delta_r2):
    if pd.isna(delta_r2):
        return "UNKNOWN"

    if delta_r2 >= 0.02:
        return "BENEFICIAL"

    if delta_r2 <= -0.02:
        return "HARMFUL"

    return "MODEL_DEPENDENT / NEUTRAL"


result["RT_VALUE_REGIME"] = (
    result["RT_DELTA_R2_XGB"]
    .apply(classify_rt_value)
)


# ============================================================
# INTERACTION VALUE CLASSIFICATION
# ============================================================

def classify_interaction(delta_r2):
    if pd.isna(delta_r2):
        return "UNKNOWN"

    if delta_r2 >= 0.02:
        return "BENEFICIAL"

    if delta_r2 <= -0.02:
        return "HARMFUL"

    return "NEGLIGIBLE"


result["INTERACTION_REGIME"] = (
    result["INTERACTION_DELTA_R2_XGB"]
    .apply(classify_interaction)
)


# ============================================================
# FEATURE DISTRIBUTION FLAGS
# ============================================================

phif_median_median = result["PHIF_MEDIAN"].median()
gr_median_median = result["GR_MEDIAN"].median()
logrt_median_median = result["LOG10_RT_MEDIAN"].median()

result["PHIF_LEVEL"] = result["PHIF_MEDIAN"].apply(
    lambda x: "HIGH" if x >= phif_median_median else "LOW"
)

result["GR_LEVEL"] = result["GR_MEDIAN"].apply(
    lambda x: "HIGH" if x >= gr_median_median else "LOW"
)

result["LOGRT_LEVEL"] = result["LOG10_RT_MEDIAN"].apply(
    lambda x: "HIGH" if x >= logrt_median_median else "LOW"
)


# ============================================================
# RELATIONSHIP DIRECTION
# ============================================================

def relationship_direction(value):
    if pd.isna(value):
        return "UNKNOWN"

    if value <= -0.20:
        return "NEGATIVE"

    if value >= 0.20:
        return "POSITIVE"

    return "WEAK / MIXED"


result["GR_PHIF_RELATIONSHIP"] = result["GR_SPEARMAN_R"].apply(
    relationship_direction
)

result["RT_PHIF_RELATIONSHIP"] = result["RT_SPEARMAN_R"].apply(
    relationship_direction
)


# ============================================================
# CROSS-WELL DIAGNOSTIC INTERPRETATION
# ============================================================

def diagnostic_interpretation(row):
    observations = []

    if row["XGB_GENERALIZATION_REGIME"] == "GOOD":
        observations.append("strong unseen-well generalization")

    elif row["XGB_GENERALIZATION_REGIME"] == "POOR":
        observations.append("poor unseen-well generalization")

    else:
        observations.append("moderate unseen-well generalization")

    if row["GR_PHIF_RELATIONSHIP"] == "NEGATIVE":
        observations.append("stable inverse GR-PHIF relationship")

    if row["RT_PHIF_RELATIONSHIP"] == "NEGATIVE":
        observations.append("inverse RT-PHIF relationship")

    elif row["RT_PHIF_RELATIONSHIP"] == "POSITIVE":
        observations.append("positive RT-PHIF relationship")

    else:
        observations.append("weak/mixed RT-PHIF relationship")

    if row["RT_VALUE_REGIME"] == "BENEFICIAL":
        observations.append("RT adds useful predictive information")

    elif row["RT_VALUE_REGIME"] == "HARMFUL":
        observations.append("RT reduces transferability")

    return "; ".join(observations)


result["DIAGNOSTIC_INTERPRETATION"] = result.apply(
    diagnostic_interpretation,
    axis=1,
)


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    by=["XGB_GENERALIZATION_REGIME", "XGB_R2"],
    ascending=[True, False],
)


# ============================================================
# SAVE
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("=" * 80)
print("CROSS-WELL FAILURE REGIME DIAGNOSTIC")
print("=" * 80)

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nGeneralization regime:")
print(
    result["XGB_GENERALIZATION_REGIME"]
    .value_counts()
    .to_string()
)

print("\nRT value regime:")
print(
    result["RT_VALUE_REGIME"]
    .value_counts()
    .to_string()
)

print("\nRT-PHIF relationship:")
print(
    result["RT_PHIF_RELATIONSHIP"]
    .value_counts()
    .to_string()
)

print("\nWell-level diagnostic:")
print(
    result[
        [
            "WELL",
            "PHIF_MEDIAN",
            "GR_MEDIAN",
            "LOG10_RT_MEDIAN",
            "GR_PHIF_RELATIONSHIP",
            "RT_PHIF_RELATIONSHIP",
            "RT_VALUE_REGIME",
            "XGB_R2",
            "XGB_GENERALIZATION_REGIME",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)