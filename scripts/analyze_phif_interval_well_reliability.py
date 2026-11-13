from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_high_quality_intervals.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_interval_well_reliability.csv"
)


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_cols = [
    "WELL",
    "THICKNESS",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "ACTUAL_PHIF_MEAN",
    "INTERVAL_MAE",
]

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DERIVED METRICS
# ============================================================

df["ACTUAL_HIGH"] = (
    df["ACTUAL_PHIF_MEAN"] >= 0.20
)

df["ABS_MEAN_ERROR"] = (
    df["PREDICTED_PHIF_MEAN"]
    - df["ACTUAL_PHIF_MEAN"]
).abs()


# ============================================================
# SCREENING RULES
# ============================================================

rules = {
    "R1_T1.0_HP80_P20": {
        "thickness": 1.0,
        "high_pct": 80,
        "predicted": 0.20,
    },
    "R2_T1.0_HP90_P22": {
        "thickness": 1.0,
        "high_pct": 90,
        "predicted": 0.22,
    },
    "R3_T2.0_HP90_P20": {
        "thickness": 2.0,
        "high_pct": 90,
        "predicted": 0.20,
    },
    "R4_T2.0_HP80_P22": {
        "thickness": 2.0,
        "high_pct": 80,
        "predicted": 0.22,
    },
}


# ============================================================
# EVALUATION
# ============================================================

all_results = []
well_results = []


for rule_name, rule in rules.items():

    mask = (
        (df["THICKNESS"] >= rule["thickness"])
        & (
            df["HIGH_SAMPLE_PCT"]
            >= rule["high_pct"]
        )
        & (
            df["PREDICTED_PHIF_MEAN"]
            >= rule["predicted"]
        )
    )

    selected = df.loc[mask].copy()

    # --------------------------------------------------------
    # Overall rule summary
    # --------------------------------------------------------

    if len(selected) > 0:

        all_results.append(
            {
                "RULE": rule_name,
                "THICKNESS_MIN": rule["thickness"],
                "HIGH_SAMPLE_PCT_MIN": rule["high_pct"],
                "PREDICTED_PHIF_MIN": rule["predicted"],
                "N_INTERVALS": len(selected),
                "N_WELLS": selected["WELL"].nunique(),
                "MEAN_THICKNESS": selected["THICKNESS"].mean(),
                "MEDIAN_THICKNESS": selected["THICKNESS"].median(),
                "MEAN_PREDICTED_PHIF": (
                    selected["PREDICTED_PHIF_MEAN"].mean()
                ),
                "MEAN_ACTUAL_PHIF": (
                    selected["ACTUAL_PHIF_MEAN"].mean()
                ),
                "ACTUAL_HIGH_RATE": (
                    selected["ACTUAL_HIGH"].mean()
                    * 100
                ),
                "MEAN_INTERVAL_MAE": (
                    selected["INTERVAL_MAE"].mean()
                ),
                "MEAN_ABS_ERROR": (
                    selected["ABS_MEAN_ERROR"].mean()
                ),
                "MEAN_BIAS": (
                    (
                        selected["PREDICTED_PHIF_MEAN"]
                        - selected["ACTUAL_PHIF_MEAN"]
                    ).mean()
                ),
            }
        )

    # --------------------------------------------------------
    # Well-level results
    # --------------------------------------------------------

    for well, group in selected.groupby("WELL"):

        well_results.append(
            {
                "RULE": rule_name,
                "WELL": well,
                "N_INTERVALS": len(group),
                "TOTAL_THICKNESS": group["THICKNESS"].sum(),
                "MEAN_THICKNESS": group["THICKNESS"].mean(),
                "MEAN_PREDICTED_PHIF": (
                    group["PREDICTED_PHIF_MEAN"].mean()
                ),
                "MEAN_ACTUAL_PHIF": (
                    group["ACTUAL_PHIF_MEAN"].mean()
                ),
                "ACTUAL_HIGH_RATE": (
                    group["ACTUAL_HIGH"].mean()
                    * 100
                ),
                "MEAN_INTERVAL_MAE": (
                    group["INTERVAL_MAE"].mean()
                ),
                "MEAN_ABS_ERROR": (
                    group["ABS_MEAN_ERROR"].mean()
                ),
                "MEAN_BIAS": (
                    (
                        group["PREDICTED_PHIF_MEAN"]
                        - group["ACTUAL_PHIF_MEAN"]
                    ).mean()
                ),
            }
        )


# ============================================================
# DATAFRAMES
# ============================================================

overall_df = pd.DataFrame(all_results)

well_df = pd.DataFrame(well_results)


# ============================================================
# WELL CONSISTENCY
# ============================================================

consistency_results = []

for rule_name in rules:

    subset = well_df[
        well_df["RULE"] == rule_name
    ].copy()

    if len(subset) == 0:
        continue

    consistency_results.append(
        {
            "RULE": rule_name,
            "WELLS_WITH_INTERVALS": len(subset),
            "WELLS_HIGH_RATE_GE_50": (
                subset["ACTUAL_HIGH_RATE"] >= 50
            ).sum(),
            "WELLS_HIGH_RATE_GE_75": (
                subset["ACTUAL_HIGH_RATE"] >= 75
            ).sum(),
            "WELLS_HIGH_RATE_GE_80": (
                subset["ACTUAL_HIGH_RATE"] >= 80
            ).sum(),
            "MEDIAN_WELL_HIGH_RATE": (
                subset["ACTUAL_HIGH_RATE"].median()
            ),
            "MIN_WELL_HIGH_RATE": (
                subset["ACTUAL_HIGH_RATE"].min()
            ),
            "MAX_WELL_HIGH_RATE": (
                subset["ACTUAL_HIGH_RATE"].max()
            ),
            "MEAN_WELL_MAE": (
                subset["MEAN_INTERVAL_MAE"].mean()
            ),
        }
    )

consistency_df = pd.DataFrame(
    consistency_results
)


# ============================================================
# PRINT
# ============================================================

print("=" * 90)
print("PHIF INTERVAL WELL-LEVEL RELIABILITY ANALYSIS")
print("=" * 90)

print(
    f"\nTotal source intervals: {len(df):,}"
)

print(
    f"Total source wells: {df['WELL'].nunique():,}"
)


print("\n--- OVERALL RULE COMPARISON ---")

if len(overall_df) > 0:

    print(
        overall_df.to_string(
            index=False
        )
    )


print("\n--- WELL-LEVEL CONSISTENCY ---")

if len(consistency_df) > 0:

    print(
        consistency_df.to_string(
            index=False
        )
    )


print("\n--- DETAILED WELL RESULTS ---")

if len(well_df) > 0:

    for rule_name in rules:

        print(
            f"\n### {rule_name}"
        )

        subset = well_df[
            well_df["RULE"] == rule_name
        ].copy()

        if len(subset) == 0:

            print(
                "No intervals selected."
            )

        else:

            print(
                subset.to_string(
                    index=False
                )
            )


# ============================================================
# DOMINANCE CHECK
# ============================================================

print("\n--- WELL DOMINANCE CHECK ---")

for rule_name in rules:

    subset = well_df[
        well_df["RULE"] == rule_name
    ].copy()

    if len(subset) == 0:
        continue

    total_intervals = subset[
        "N_INTERVALS"
    ].sum()

    max_intervals = subset[
        "N_INTERVALS"
    ].max()

    dominant_well = subset.loc[
        subset["N_INTERVALS"].idxmax(),
        "WELL",
    ]

    dominance_pct = (
        max_intervals
        / total_intervals
        * 100
    )

    print(
        f"{rule_name}: "
        f"{dominant_well} contributes "
        f"{max_intervals}/{total_intervals} "
        f"intervals "
        f"({dominance_pct:.1f}%)"
    )


# ============================================================
# SAVE
# ============================================================

if len(well_df) > 0:

    well_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

else:

    # Still create an empty file with schema.
    pd.DataFrame(
        columns=[
            "RULE",
            "WELL",
            "N_INTERVALS",
            "TOTAL_THICKNESS",
            "MEAN_THICKNESS",
            "MEAN_PREDICTED_PHIF",
            "MEAN_ACTUAL_PHIF",
            "ACTUAL_HIGH_RATE",
            "MEAN_INTERVAL_MAE",
            "MEAN_ABS_ERROR",
            "MEAN_BIAS",
        ]
    ).to_csv(
        OUTPUT_FILE,
        index=False,
    )


print("\nSaved:")
print(OUTPUT_FILE)