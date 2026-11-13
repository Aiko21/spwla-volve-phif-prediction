from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RELATIONSHIP_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "phif_relationship_by_well.csv"
)

GENERALIZATION_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "well_generalization_diagnostic.csv"
)

RT_INCREMENTAL_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "rt_incremental_value_by_well.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "rt_regime_by_well.csv"
)


# ============================================================
# LOAD INPUTS
# ============================================================

relationship = pd.read_csv(RELATIONSHIP_PATH)
generalization = pd.read_csv(GENERALIZATION_PATH)
incremental = pd.read_csv(RT_INCREMENTAL_PATH)


# ============================================================
# SELECT WELL-LEVEL FEATURES
# ============================================================

relationship_cols = [
    "WELL",
    "PHIF_MEDIAN",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
]

relationship_selected = relationship[relationship_cols].copy()


generalization_cols = [
    "WELL",
    "GR_MEDIAN",
    "RT_MEDIAN",
    "LOG10_RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_STD",
]

generalization_selected = generalization[generalization_cols].copy()


# ============================================================
# PIVOT INCREMENTAL MODEL RESULTS
# ============================================================

incremental_selected = incremental[
    [
        "HELD_OUT_WELL",
        "MODEL",
        "DELTA_MAE",
        "DELTA_RMSE",
        "DELTA_R2",
    ]
].copy()

incremental_selected = incremental_selected.rename(
    columns={"HELD_OUT_WELL": "WELL"}
)


linear = incremental_selected[
    incremental_selected["MODEL"] == "LINEAR"
].copy()

linear = linear.rename(
    columns={
        "DELTA_MAE": "LINEAR_DELTA_MAE",
        "DELTA_RMSE": "LINEAR_DELTA_RMSE",
        "DELTA_R2": "LINEAR_DELTA_R2",
    }
)

linear = linear.drop(columns=["MODEL"])


xgb = incremental_selected[
    incremental_selected["MODEL"] == "XGBOOST"
].copy()

xgb = xgb.rename(
    columns={
        "DELTA_MAE": "XGB_DELTA_MAE",
        "DELTA_RMSE": "XGB_DELTA_RMSE",
        "DELTA_R2": "XGB_DELTA_R2",
    }
)

xgb = xgb.drop(columns=["MODEL"])


# ============================================================
# MERGE
# ============================================================

df = relationship_selected.merge(
    generalization_selected,
    on="WELL",
    how="left",
)

df = df.merge(
    linear,
    on="WELL",
    how="left",
)

df = df.merge(
    xgb,
    on="WELL",
    how="left",
)


# ============================================================
# DESCRIPTIVE RT REGIME CLASSIFICATION
# ============================================================
#
# IMPORTANT:
# This is NOT a machine-learning classification.
# It is only a descriptive diagnostic based on
# incremental R2 from GR -> GR+RT.
#
# Positive delta R2  = RT helped
# Negative delta R2  = RT hurt
# Near zero          = RT approximately neutral
#
# Threshold = +/- 0.02
# This threshold is intentionally descriptive, not statistical.
# ============================================================

def classify_delta_r2(value):
    if pd.isna(value):
        return "UNKNOWN"

    if value > 0.02:
        return "BENEFICIAL"

    if value < -0.02:
        return "HARMFUL"

    return "NEUTRAL"


df["LINEAR_RT_REGIME"] = df["LINEAR_DELTA_R2"].apply(
    classify_delta_r2
)

df["XGB_RT_REGIME"] = df["XGB_DELTA_R2"].apply(
    classify_delta_r2
)


# ============================================================
# COMBINED DESCRIPTIVE REGIME
# ============================================================

def combined_regime(row):
    linear = row["LINEAR_RT_REGIME"]
    xgb = row["XGB_RT_REGIME"]

    if linear == "BENEFICIAL" and xgb == "BENEFICIAL":
        return "CONSISTENTLY_BENEFICIAL"

    if linear == "HARMFUL" and xgb == "HARMFUL":
        return "CONSISTENTLY_HARMFUL"

    if linear == "NEUTRAL" and xgb == "NEUTRAL":
        return "CONSISTENTLY_NEUTRAL"

    return "MODEL_DEPENDENT"


df["COMBINED_RT_REGIME"] = df.apply(
    combined_regime,
    axis=1,
)


# ============================================================
# OUTPUT ORDER
# ============================================================

output_cols = [
    "WELL",
    "PHIF_MEDIAN",
    "GR_MEDIAN",
    "RT_MEDIAN",
    "LOG10_RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LINEAR_DELTA_MAE",
    "LINEAR_DELTA_RMSE",
    "LINEAR_DELTA_R2",
    "XGB_DELTA_MAE",
    "XGB_DELTA_RMSE",
    "XGB_DELTA_R2",
    "LINEAR_RT_REGIME",
    "XGB_RT_REGIME",
    "COMBINED_RT_REGIME",
]

df = df[output_cols].sort_values("XGB_DELTA_R2", ascending=False)


# ============================================================
# PRINT
# ============================================================

print("=" * 100)
print("RT REGIME ANALYSIS BY WELL")
print("=" * 100)
print()

print(
    df[
        [
            "WELL",
            "PHIF_MEDIAN",
            "GR_MEDIAN",
            "RT_MEDIAN",
            "LOG10_RT_MEDIAN",
            "GR_SPEARMAN_R",
            "RT_SPEARMAN_R",
            "LINEAR_DELTA_R2",
            "XGB_DELTA_R2",
            "COMBINED_RT_REGIME",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("REGIME COUNTS")
print("=" * 100)
print()

print("Linear:")
print(
    df["LINEAR_RT_REGIME"]
    .value_counts()
    .to_string()
)

print()
print("XGBoost:")
print(
    df["XGB_RT_REGIME"]
    .value_counts()
    .to_string()
)

print()
print("Combined:")
print(
    df["COMBINED_RT_REGIME"]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("RT RELATIONSHIP VS INCREMENTAL VALUE")
print("=" * 100)
print()

print(
    df[
        [
            "WELL",
            "RT_SPEARMAN_R",
            "LINEAR_DELTA_R2",
            "XGB_DELTA_R2",
            "COMBINED_RT_REGIME",
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)
print(OUTPUT_PATH)
print()
print("Done.")