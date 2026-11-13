from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

relationship_path = (
    ROOT / "reports" / "data_quality" / "phif_relationship_by_well.csv"
)

feature_path = (
    ROOT / "reports" / "data_quality" / "feature_distribution_by_well.csv"
)

cv_path = (
    ROOT / "reports" / "models" / "phif_group_cv_results.csv"
)

relationship = pd.read_csv(relationship_path)
features = pd.read_csv(feature_path)
cv = pd.read_csv(cv_path)

# ---------------------------------------------------------
# Extract PHIF / relationship information
# ---------------------------------------------------------

rel_cols = [
    "WELL",
    "PHIF_MEAN",
    "PHIF_STD",
    "PHIF_MEDIAN",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LOGRT_SPEARMAN_R",
]

relationship = relationship[rel_cols]

# ---------------------------------------------------------
# Feature distribution: keep median/std
# ---------------------------------------------------------

feature_summary = features[
    features["FEATURE"].isin(["GR", "RT", "LOG10_RT"])
].pivot(
    index="WELL",
    columns="FEATURE",
    values=["MEDIAN", "STD"]
)

feature_summary.columns = [
    f"{feature}_{stat}"
    for stat, feature in feature_summary.columns
]

feature_summary = feature_summary.reset_index()

# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

cv_summary = cv[
    [
        "HELD_OUT_WELL",
        "LINEAR_MAE",
        "LINEAR_R2",
        "XGB_MAE",
        "XGB_R2",
    ]
].rename(columns={"HELD_OUT_WELL": "WELL"})

# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

diagnostic = (
    relationship
    .merge(feature_summary, on="WELL", how="left")
    .merge(cv_summary, on="WELL", how="left")
)

# ---------------------------------------------------------
# Reorder
# ---------------------------------------------------------

columns = [
    "WELL",
    "PHIF_MEAN",
    "PHIF_STD",
    "PHIF_MEDIAN",
    "GR_MEDIAN",
    "GR_STD",
    "RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_MEDIAN",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LOGRT_SPEARMAN_R",
    "LINEAR_MAE",
    "LINEAR_R2",
    "XGB_MAE",
    "XGB_R2",
]

diagnostic = diagnostic[columns]

print("=" * 78)
print("WELL GENERALIZATION DIAGNOSTIC MATRIX")
print("=" * 78)
print()
print(diagnostic.to_string(index=False))

# ---------------------------------------------------------
# Simple correlations with XGB performance
# ---------------------------------------------------------

numeric_cols = [
    "PHIF_MEDIAN",
    "PHIF_STD",
    "GR_MEDIAN",
    "GR_STD",
    "RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_MEDIAN",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LOGRT_SPEARMAN_R",
]

print()
print("=" * 78)
print("CORRELATION WITH XGBOOST R2")
print("=" * 78)

for col in numeric_cols:
    corr = diagnostic[col].corr(diagnostic["XGB_R2"])
    print(f"{col:25s}: {corr:+.4f}")

print()
print("=" * 78)
print("CORRELATION WITH XGBOOST MAE")
print("=" * 78)

for col in numeric_cols:
    corr = diagnostic[col].corr(diagnostic["XGB_MAE"])
    print(f"{col:25s}: {corr:+.4f}")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output_path = (
    ROOT / "reports" / "data_quality" /
    "well_generalization_diagnostic.csv"
)

diagnostic.to_csv(output_path, index=False)

print()
print("=" * 78)
print(f"OUTPUT")
print("=" * 78)
print(output_path)
print()
print("Done.")