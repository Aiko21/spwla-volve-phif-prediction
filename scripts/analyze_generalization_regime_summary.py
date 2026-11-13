from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "crosswell_failure_regimes.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "generalization_regime_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("GENERALIZATION REGIME SUMMARY")
print("=" * 80)

print(f"\nInput:")
print(INPUT_FILE)

print(f"\nWells: {len(df)}")


# ============================================================
# NUMERIC SUMMARY BY GENERALIZATION REGIME
# ============================================================

numeric_cols = [
    "PHIF_MEDIAN",
    "GR_MEDIAN",
    "RT_MEDIAN",
    "LOG10_RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "RT_DELTA_MAE_XGB",
    "RT_DELTA_RMSE_XGB",
    "RT_DELTA_R2_XGB",
    "INTERACTION_DELTA_MAE_XGB",
    "INTERACTION_DELTA_RMSE_XGB",
    "INTERACTION_DELTA_R2_XGB",
    "LINEAR_R2",
    "XGB_R2",
]

summary = (
    df.groupby("XGB_GENERALIZATION_REGIME")[numeric_cols]
    .agg(["mean", "median"])
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("NUMERIC SUMMARY BY GENERALIZATION REGIME")
print("=" * 80)

print(summary.to_string())


# ============================================================
# RELATIONSHIP COUNTS
# ============================================================

print("\n" + "=" * 80)
print("RT-PHIF RELATIONSHIP BY GENERALIZATION REGIME")
print("=" * 80)

relationship_table = pd.crosstab(
    df["XGB_GENERALIZATION_REGIME"],
    df["RT_PHIF_RELATIONSHIP"],
)

print(relationship_table.to_string())


print("\n" + "=" * 80)
print("RT VALUE REGIME BY GENERALIZATION REGIME")
print("=" * 80)

rt_value_table = pd.crosstab(
    df["XGB_GENERALIZATION_REGIME"],
    df["RT_VALUE_REGIME"],
)

print(rt_value_table.to_string())


print("\n" + "=" * 80)
print("INTERACTION REGIME BY GENERALIZATION REGIME")
print("=" * 80)

interaction_table = pd.crosstab(
    df["XGB_GENERALIZATION_REGIME"],
    df["INTERACTION_REGIME"],
)

print(interaction_table.to_string())


# ============================================================
# WELL LIST
# ============================================================

print("\n" + "=" * 80)
print("WELLS BY GENERALIZATION REGIME")
print("=" * 80)

for regime in ["GOOD", "MODERATE", "POOR"]:
    subset = df[
        df["XGB_GENERALIZATION_REGIME"] == regime
    ].copy()

    print(f"\n{regime} ({len(subset)} wells):")

    if subset.empty:
        print("  None")
        continue

    for _, row in subset.sort_values("XGB_R2", ascending=False).iterrows():
        print(
            f"  {row['WELL']}: "
            f"R2={row['XGB_R2']:.4f}, "
            f"PHIF_med={row['PHIF_MEDIAN']:.4f}, "
            f"GR_med={row['GR_MEDIAN']:.2f}, "
            f"logRT_med={row['LOG10_RT_MEDIAN']:.4f}, "
            f"RT_rel={row['RT_PHIF_RELATIONSHIP']}, "
            f"RT_value={row['RT_VALUE_REGIME']}"
        )


# ============================================================
# DESCRIPTIVE DIFFERENCE: GOOD vs POOR
# ============================================================

good = df[
    df["XGB_GENERALIZATION_REGIME"] == "GOOD"
]

poor = df[
    df["XGB_GENERALIZATION_REGIME"] == "POOR"
]

comparison_rows = []

for col in numeric_cols:
    good_mean = good[col].mean()
    poor_mean = poor[col].mean()

    comparison_rows.append(
        {
            "FEATURE": col,
            "GOOD_MEAN": good_mean,
            "POOR_MEAN": poor_mean,
            "GOOD_MINUS_POOR": good_mean - poor_mean,
        }
    )

comparison = pd.DataFrame(comparison_rows)

print("\n" + "=" * 80)
print("GOOD vs POOR — DESCRIPTIVE DIFFERENCES")
print("=" * 80)

print(
    comparison
    .sort_values(
        "GOOD_MINUS_POOR",
        key=lambda x: x.abs(),
        ascending=False,
    )
    .to_string(index=False)
)


# ============================================================
# SAVE WELL-LEVEL SUMMARY
# ============================================================

# Keep the original well-level observations together with
# regime labels. This makes the output easy to reuse later.

output_cols = [
    "WELL",
    "XGB_GENERALIZATION_REGIME",
    "XGB_R2",
    "LINEAR_R2",
    "PHIF_MEDIAN",
    "GR_MEDIAN",
    "RT_MEDIAN",
    "LOG10_RT_MEDIAN",
    "RT_STD",
    "LOG10_RT_STD",
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "GR_PHIF_RELATIONSHIP",
    "RT_PHIF_RELATIONSHIP",
    "RT_VALUE_REGIME",
    "INTERACTION_REGIME",
    "RT_DELTA_MAE_XGB",
    "RT_DELTA_RMSE_XGB",
    "RT_DELTA_R2_XGB",
    "INTERACTION_DELTA_MAE_XGB",
    "INTERACTION_DELTA_RMSE_XGB",
    "INTERACTION_DELTA_R2_XGB",
    "DIAGNOSTIC_INTERPRETATION",
]

output = df[output_cols].copy()

output = output.sort_values(
    ["XGB_GENERALIZATION_REGIME", "XGB_R2"],
    ascending=[True, False],
)

output.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("OUTPUT SAVED")
print("=" * 80)

print(OUTPUT_FILE)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)