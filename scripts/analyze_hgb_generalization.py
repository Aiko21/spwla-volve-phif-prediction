from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HGB_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_cv_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "hgb_generalization_diagnostic.csv"
)


# ============================================================
# LOAD
# ============================================================

hgb = pd.read_csv(HGB_FILE)

hgb.columns = hgb.columns.str.strip()


# ============================================================
# SELECT MODEL RESULTS
# ============================================================

linear = hgb[
    hgb["MODEL"].str.upper().eq("LINEAR")
].copy()

xgb = hgb[
    hgb["MODEL"].str.upper().eq("XGBOOST")
].copy()

hgb_model = hgb[
    hgb["MODEL"].str.upper().eq("HGB")
].copy()


# ============================================================
# RENAME
# ============================================================

linear = linear[
    [
        "HELD_OUT_WELL",
        "MAE",
        "RMSE",
        "R2",
    ]
].rename(
    columns={
        "HELD_OUT_WELL": "WELL",
        "MAE": "LINEAR_MAE",
        "RMSE": "LINEAR_RMSE",
        "R2": "LINEAR_R2",
    }
)

xgb = xgb[
    [
        "HELD_OUT_WELL",
        "MAE",
        "RMSE",
        "R2",
    ]
].rename(
    columns={
        "HELD_OUT_WELL": "WELL",
        "MAE": "XGB_MAE",
        "RMSE": "XGB_RMSE",
        "R2": "XGB_R2",
    }
)

hgb_model = hgb_model[
    [
        "HELD_OUT_WELL",
        "MAE",
        "RMSE",
        "R2",
    ]
].rename(
    columns={
        "HELD_OUT_WELL": "WELL",
        "MAE": "HGB_MAE",
        "RMSE": "HGB_RMSE",
        "R2": "HGB_R2",
    }
)


# ============================================================
# MERGE
# ============================================================

result = linear.merge(
    xgb,
    on="WELL",
    how="inner",
)

result = result.merge(
    hgb_model,
    on="WELL",
    how="inner",
)


# ============================================================
# MODEL DIFFERENCES
# ============================================================

# Positive R2 delta means HGB is better.

result["HGB_VS_XGB_DELTA_R2"] = (
    result["HGB_R2"]
    - result["XGB_R2"]
)

result["HGB_VS_LINEAR_DELTA_R2"] = (
    result["HGB_R2"]
    - result["LINEAR_R2"]
)


# Positive MAE improvement means HGB is better.
# Therefore:
#
# HGB_MAE_IMPROVEMENT_VS_XGB
# = XGB_MAE - HGB_MAE

result["HGB_MAE_IMPROVEMENT_VS_XGB"] = (
    result["XGB_MAE"]
    - result["HGB_MAE"]
)

result["HGB_MAE_IMPROVEMENT_VS_LINEAR"] = (
    result["LINEAR_MAE"]
    - result["HGB_MAE"]
)


# ============================================================
# CLASSIFICATION
# ============================================================

# Threshold is intentionally descriptive.
# +/- 0.02 R2 is used as a practical diagnostic threshold,
# not as a statistical significance threshold.

def classify_r2_delta(delta):
    if delta >= 0.02:
        return "HGB_IMPROVES"

    if delta <= -0.02:
        return "HGB_DEGRADES"

    return "HGB_SIMILAR"


result["HGB_VS_XGB_REGIME"] = (
    result["HGB_VS_XGB_DELTA_R2"]
    .apply(classify_r2_delta)
)

result["HGB_VS_LINEAR_REGIME"] = (
    result["HGB_VS_LINEAR_DELTA_R2"]
    .apply(classify_r2_delta)
)


# ============================================================
# OVERALL BEST MODEL PER WELL
# ============================================================

def best_model(row):

    scores = {
        "LINEAR": row["LINEAR_R2"],
        "XGBOOST": row["XGB_R2"],
        "HGB": row["HGB_R2"],
    }

    return max(
        scores,
        key=scores.get,
    )


result["BEST_MODEL_BY_R2"] = result.apply(
    best_model,
    axis=1,
)


# ============================================================
# HGB PERFORMANCE CATEGORY
# ============================================================

def hgb_performance(r2):

    if r2 >= 0.20:
        return "GOOD"

    if r2 >= 0.00:
        return "MODERATE"

    return "POOR"


result["HGB_GENERALIZATION_REGIME"] = (
    result["HGB_R2"]
    .apply(hgb_performance)
)


# ============================================================
# SORT
# ============================================================

result = result.sort_values(
    "HGB_R2",
    ascending=False,
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# CONSOLE
# ============================================================

print("=" * 80)
print("HGB GENERALIZATION DIAGNOSTIC")
print("=" * 80)

print("\nInput:")
print(HGB_FILE)

print("\nOutput:")
print(OUTPUT_FILE)


# ============================================================
# WELL-LEVEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON BY WELL")
print("=" * 80)

display_cols = [
    "WELL",
    "LINEAR_MAE",
    "XGB_MAE",
    "HGB_MAE",
    "LINEAR_R2",
    "XGB_R2",
    "HGB_R2",
    "HGB_VS_XGB_DELTA_R2",
    "HGB_VS_LINEAR_DELTA_R2",
    "BEST_MODEL_BY_R2",
    "HGB_GENERALIZATION_REGIME",
]

print(
    result[
        display_cols
    ].to_string(index=False)
)


# ============================================================
# HGB VS XGB
# ============================================================

print("\n" + "=" * 80)
print("HGB VS XGBOOST")
print("=" * 80)

print(
    result[
        [
            "WELL",
            "HGB_VS_XGB_DELTA_R2",
            "HGB_MAE_IMPROVEMENT_VS_XGB",
            "HGB_VS_XGB_REGIME",
        ]
    ].to_string(index=False)
)

print("\nCounts:")

print(
    result[
        "HGB_VS_XGB_REGIME"
    ].value_counts().to_string()
)


# ============================================================
# HGB VS LINEAR
# ============================================================

print("\n" + "=" * 80)
print("HGB VS LINEAR")
print("=" * 80)

print(
    result[
        [
            "WELL",
            "HGB_VS_LINEAR_DELTA_R2",
            "HGB_MAE_IMPROVEMENT_VS_LINEAR",
            "HGB_VS_LINEAR_REGIME",
        ]
    ].to_string(index=False)
)

print("\nCounts:")

print(
    result[
        "HGB_VS_LINEAR_REGIME"
    ].value_counts().to_string()
)


# ============================================================
# BEST MODEL COUNTS
# ============================================================

print("\n" + "=" * 80)
print("BEST MODEL BY WELL — R2")
print("=" * 80)

print(
    result[
        "BEST_MODEL_BY_R2"
    ].value_counts().to_string()
)


# ============================================================
# HGB GENERALIZATION
# ============================================================

print("\n" + "=" * 80)
print("HGB GENERALIZATION REGIME")
print("=" * 80)

print(
    result[
        "HGB_GENERALIZATION_REGIME"
    ].value_counts().to_string()
)


# ============================================================
# AGGREGATE IMPROVEMENT
# ============================================================

print("\n" + "=" * 80)
print("AGGREGATE HGB IMPROVEMENT")
print("=" * 80)

print(
    f"Mean ΔR2 vs XGBoost  : "
    f"{result['HGB_VS_XGB_DELTA_R2'].mean():.6f}"
)

print(
    f"Mean ΔR2 vs Linear   : "
    f"{result['HGB_VS_LINEAR_DELTA_R2'].mean():.6f}"
)

print(
    f"Mean MAE improvement "
    f"vs XGBoost          : "
    f"{result['HGB_MAE_IMPROVEMENT_VS_XGB'].mean():.6f}"
)

print(
    f"Mean MAE improvement "
    f"vs Linear           : "
    f"{result['HGB_MAE_IMPROVEMENT_VS_LINEAR'].mean():.6f}"
)


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)