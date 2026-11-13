from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HGB_RESULTS = ROOT / "reports" / "models" / "phif_hgb_cv_results.csv"
CROSSWELL = ROOT / "reports" / "data_quality" / "crosswell_failure_regimes.csv"
COVARIATE = ROOT / "reports" / "data_quality" / "covariate_relationship_shift.csv"
RT_REGIME = ROOT / "reports" / "data_quality" / "rt_incremental_value_by_well.csv"

OUTPUT = ROOT / "reports" / "data_quality" / "hgb_failure_regimes.csv"


# ============================================================
# HELPERS
# ============================================================

def normalize_well_name(value):
    value = str(value).strip()

    if value.startswith("15_9-"):
        value = value.replace("15_9-", "", 1)

    return value


def detect_well_column(df):
    candidates = [
        "HELD_OUT_WELL",
        "WELL",
        "well",
        "Well",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not identify well column. "
        f"Available columns: {list(df.columns)}"
    )


def make_unique_per_well(df, name):
    """
    Ensure one row per well before merging.
    """
    before = len(df)

    df = df.drop_duplicates(subset=["WELL"]).copy()

    after = len(df)

    if before != after:
        print(
            f"[INFO] {name}: removed "
            f"{before - after} duplicate well rows."
        )

    return df


# ============================================================
# START
# ============================================================

print("=" * 80)
print("HGB FAILURE REGIME ANALYSIS")
print("=" * 80)


# ============================================================
# HGB
# ============================================================

print("\nLoading HGB results...")

hgb = pd.read_csv(HGB_RESULTS)

hgb_col = detect_well_column(hgb)
hgb = hgb.rename(columns={hgb_col: "WELL"})

if "MODEL" in hgb.columns:
    hgb = hgb[
        hgb["MODEL"].astype(str).str.upper() == "HGB"
    ].copy()

hgb["WELL"] = hgb["WELL"].map(normalize_well_name)

required = ["WELL", "MAE", "RMSE", "R2"]

missing = [c for c in required if c not in hgb.columns]

if missing:
    raise ValueError(
        f"HGB result missing columns: {missing}"
    )

hgb = hgb[required].copy()

hgb = hgb.rename(
    columns={
        "MAE": "HGB_MAE",
        "RMSE": "HGB_RMSE",
        "R2": "HGB_R2",
    }
)

hgb = make_unique_per_well(hgb, "HGB")


# ============================================================
# CROSSWELL
# ============================================================

print("\nLoading crosswell diagnostics...")

crosswell = pd.read_csv(CROSSWELL)

print("Crosswell columns:")
print(list(crosswell.columns))

cross_col = detect_well_column(crosswell)
crosswell = crosswell.rename(columns={cross_col: "WELL"})

crosswell["WELL"] = crosswell["WELL"].map(normalize_well_name)

# Prefer XGBoost row if MODEL exists
if "MODEL" in crosswell.columns:

    model_values = (
        crosswell["MODEL"]
        .astype(str)
        .str.upper()
    )

    if "XGBOOST" in set(model_values):

        crosswell = crosswell[
            model_values == "XGBOOST"
        ].copy()

        print("[INFO] Crosswell filtered to XGBOOST rows.")

crosswell = make_unique_per_well(
    crosswell,
    "Crosswell",
)


# ============================================================
# COVARIATE / RELATIONSHIP SHIFT
# ============================================================

print("\nLoading covariate/relationship shift diagnostics...")

covariate = pd.read_csv(COVARIATE)

print("Covariate columns:")
print(list(covariate.columns))

cov_col = detect_well_column(covariate)
covariate = covariate.rename(columns={cov_col: "WELL"})

covariate["WELL"] = covariate["WELL"].map(normalize_well_name)

covariate = make_unique_per_well(
    covariate,
    "Covariate",
)


# ============================================================
# RT REGIME
# ============================================================

print("\nLoading RT diagnostics...")

rt = pd.read_csv(RT_REGIME)

print("RT columns:")
print(list(rt.columns))

rt_col = detect_well_column(rt)
rt = rt.rename(columns={rt_col: "WELL"})

rt["WELL"] = rt["WELL"].map(normalize_well_name)

# If MODEL exists, retain XGBoost rows where available
if "MODEL" in rt.columns:

    model_values = (
        rt["MODEL"]
        .astype(str)
        .str.upper()
    )

    if "XGBOOST" in set(model_values):

        rt = rt[
            model_values == "XGBOOST"
        ].copy()

        print("[INFO] RT diagnostics filtered to XGBOOST rows.")

rt = make_unique_per_well(
    rt,
    "RT",
)


# ============================================================
# SELECT COLUMNS
# ============================================================

crosswell_keep = ["WELL"]

for col in [
    "XGB_R2",
    "GENERALIZATION_REGIME",
    "RT_VALUE_REGIME",
    "RT_RELATIONSHIP_REGIME",
]:
    if col in crosswell.columns:
        crosswell_keep.append(col)

crosswell = crosswell[crosswell_keep].copy()


covariate_keep = ["WELL"]

for col in [
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_PHIF_SPEARMAN",
    "REFERENCE_GR_PHIF_SPEARMAN",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_PHIF_SPEARMAN",
    "REFERENCE_LOG10_RT_PHIF_SPEARMAN",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
]:
    if col in covariate.columns:
        covariate_keep.append(col)

covariate = covariate[covariate_keep].copy()


rt_keep = ["WELL"]

for col in [
    "DELTA_MAE",
    "DELTA_RMSE",
    "DELTA_R2",
    "MAE_IMPROVED",
    "RMSE_IMPROVED",
    "R2_IMPROVED",
]:
    if col in rt.columns:
        rt_keep.append(col)

rt = rt[rt_keep].copy()


# ============================================================
# MERGE
# ============================================================

diagnostic = hgb.copy()

diagnostic = diagnostic.merge(
    crosswell,
    on="WELL",
    how="left",
)

diagnostic = diagnostic.merge(
    covariate,
    on="WELL",
    how="left",
)

diagnostic = diagnostic.merge(
    rt,
    on="WELL",
    how="left",
)

# Remove duplicated column names created during diagnostic merges.
diagnostic = diagnostic.loc[
    :,
    ~diagnostic.columns.duplicated()
].copy()


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

if diagnostic["WELL"].duplicated().any():

    duplicates = diagnostic[
        diagnostic["WELL"].duplicated(
            keep=False
        )
    ]

    print("\nERROR: Duplicate wells remain:")
    print(duplicates)

    raise ValueError(
        "Diagnostic must contain exactly one row per well."
    )


if len(diagnostic) != 9:

    raise ValueError(
        f"Expected 9 wells, found {len(diagnostic)}."
    )


# ============================================================
# HGB GENERALIZATION REGIME
# ============================================================

def hgb_regime(r2):

    if r2 >= 0.20:
        return "GOOD"

    elif r2 >= 0:
        return "MODERATE"

    return "POOR"


diagnostic["HGB_GENERALIZATION_REGIME"] = (
    diagnostic["HGB_R2"].apply(hgb_regime)
)


# ============================================================
# SORT
# ============================================================

diagnostic = diagnostic.sort_values(
    "HGB_R2",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

diagnostic.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# MAIN TABLE
# ============================================================

print("\n" + "=" * 80)
print("HGB GENERALIZATION BY WELL")
print("=" * 80)

display_cols = [
    "WELL",
    "HGB_MAE",
    "HGB_RMSE",
    "HGB_R2",
    "HGB_GENERALIZATION_REGIME",
]

for col in [
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
    "DELTA_R2",
]:

    if col in diagnostic.columns:
        display_cols.append(col)


print(
    diagnostic[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# REGIME COUNTS
# ============================================================

print("\n" + "=" * 80)
print("HGB REGIME COUNTS")
print("=" * 80)

print(
    diagnostic[
        "HGB_GENERALIZATION_REGIME"
    ].value_counts().to_string()
)


# ============================================================
# REGIME SUMMARY
# ============================================================

numeric_candidates = [
    "HGB_R2",
    "HGB_MAE",
    "HGB_RMSE",
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
    "DELTA_R2",
]

numeric_cols = [
    c
    for c in numeric_candidates
    if c in diagnostic.columns
]


summary = (
    diagnostic
    .groupby("HGB_GENERALIZATION_REGIME")[
        numeric_cols
    ]
    .mean()
    .reindex(
        ["GOOD", "MODERATE", "POOR"]
    )
)


print("\n" + "=" * 80)
print("MEAN DIAGNOSTIC VALUES BY HGB REGIME")
print("=" * 80)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SPEARMAN CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATORY SPEARMAN CORRELATION WITH HGB R2")
print("=" * 80)

correlation_candidates = [
    "GR_MEDIAN_SHIFT",
    "GR_IQR_SHIFT",
    "LOG10_RT_MEDIAN_SHIFT",
    "LOG10_RT_IQR_SHIFT",
    "GR_RELATIONSHIP_SHIFT",
    "LOG10_RT_RELATIONSHIP_SHIFT",
    "COVARIATE_SHIFT_INDEX",
    "RELATIONSHIP_SHIFT_INDEX",
    "TOTAL_DIAGNOSTIC_SHIFT",
    "DELTA_R2",
]

corr_rows = []

for col in correlation_candidates:

    if col not in diagnostic.columns:
        continue

    temp = diagnostic[
        ["HGB_R2", col]
    ].dropna()

    if len(temp) < 3:
        continue

    corr = temp["HGB_R2"].corr(
        temp[col],
        method="spearman",
    )

    corr_rows.append(
        {
            "FEATURE": col,
            "N": len(temp),
            "SPEARMAN_R": corr,
        }
    )


corr_df = pd.DataFrame(corr_rows)

if not corr_df.empty:

    corr_df = corr_df.sort_values(
        "SPEARMAN_R",
        ascending=False,
    )

    print(
        corr_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# ============================================================
# FAILURE WELLS
# ============================================================

print("\n" + "=" * 80)
print("HGB FAILURE WELLS")
print("=" * 80)

failure = diagnostic[
    diagnostic["HGB_GENERALIZATION_REGIME"]
    == "POOR"
]

print(
    failure[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SUCCESS WELLS
# ============================================================

print("\n" + "=" * 80)
print("HGB SUCCESS WELLS")
print("=" * 80)

success = diagnostic[
    diagnostic["HGB_GENERALIZATION_REGIME"]
    == "GOOD"
]

print(
    success[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(f"Unique wells : {diagnostic['WELL'].nunique()}")
print(f"Rows         : {len(diagnostic)}")

print("\nOutput:")
print(OUTPUT)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
