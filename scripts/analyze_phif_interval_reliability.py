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
    / "phif_interval_reliability_analysis.csv"
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
    "TOP_DEPTH",
    "BASE_DEPTH",
    "THICKNESS",
    "N_HIGH_SAMPLES",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "PREDICTED_PHIF_MAX",
    "PREDICTED_PHIF_MIN",
    "ACTUAL_PHIF_MEAN",
    "ACTUAL_PHIF_MAX",
    "INTERVAL_MAE",
]

missing = [
    col for col in required_cols
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# DERIVED VALIDATION METRICS
# ============================================================

# Actual PHIF >= 0.20
df["ACTUAL_HIGH_PHIF"] = (
    df["ACTUAL_PHIF_MEAN"] >= 0.20
)

# Prediction error direction
df["PREDICTION_BIAS"] = (
    df["PREDICTED_PHIF_MEAN"]
    - df["ACTUAL_PHIF_MEAN"]
)

# Absolute prediction error at interval mean
df["ABS_MEAN_ERROR"] = (
    df["PREDICTION_BIAS"].abs()
)


# ============================================================
# SCREENING FUNCTION
# ============================================================

def evaluate_rule(
    data,
    thickness_min,
    high_pct_min,
    predicted_mean_min,
    mae_max=None,
):
    """
    Evaluate an interval screening rule.

    NOTE:
    INTERVAL_MAE is included only for retrospective validation.
    It must NOT be used as a deployment-time confidence feature.
    """

    mask = (
        (data["THICKNESS"] >= thickness_min)
        & (
            data["HIGH_SAMPLE_PCT"]
            >= high_pct_min
        )
        & (
            data["PREDICTED_PHIF_MEAN"]
            >= predicted_mean_min
        )
    )

    if mae_max is not None:
        mask &= (
            data["INTERVAL_MAE"]
            <= mae_max
        )

    selected = data.loc[mask].copy()

    if len(selected) == 0:
        return {
            "N_INTERVALS": 0,
            "PCT_ALL_INTERVALS": 0.0,
            "N_WELLS": 0,
            "MEAN_THICKNESS": np.nan,
            "MEDIAN_THICKNESS": np.nan,
            "MEAN_PREDICTED_PHIF": np.nan,
            "MEAN_ACTUAL_PHIF": np.nan,
            "MEAN_INTERVAL_MAE": np.nan,
            "ACTUAL_HIGH_RATE": np.nan,
            "MEAN_ABS_ERROR": np.nan,
            "MEAN_BIAS": np.nan,
        }

    return {
        "N_INTERVALS": len(selected),
        "PCT_ALL_INTERVALS": (
            len(selected)
            / len(data)
            * 100
        ),
        "N_WELLS": selected["WELL"].nunique(),
        "MEAN_THICKNESS": selected["THICKNESS"].mean(),
        "MEDIAN_THICKNESS": selected["THICKNESS"].median(),
        "MEAN_PREDICTED_PHIF": (
            selected["PREDICTED_PHIF_MEAN"].mean()
        ),
        "MEAN_ACTUAL_PHIF": (
            selected["ACTUAL_PHIF_MEAN"].mean()
        ),
        "MEAN_INTERVAL_MAE": (
            selected["INTERVAL_MAE"].mean()
        ),
        "ACTUAL_HIGH_RATE": (
            selected["ACTUAL_HIGH_PHIF"].mean()
            * 100
        ),
        "MEAN_ABS_ERROR": (
            selected["ABS_MEAN_ERROR"].mean()
        ),
        "MEAN_BIAS": (
            selected["PREDICTION_BIAS"].mean()
        ),
    }


# ============================================================
# BASELINE
# ============================================================

print("=" * 90)
print("PHIF INTERVAL RETROSPECTIVE RELIABILITY ANALYSIS")
print("=" * 90)

print(f"\nInput: {INPUT_FILE}")

print(f"\nTotal intervals: {len(df):,}")
print(f"Total wells: {df['WELL'].nunique():,}")

print("\n--- OVERALL INTERVAL QUALITY ---")

print(
    f"Mean predicted PHIF: "
    f"{df['PREDICTED_PHIF_MEAN'].mean():.4f}"
)

print(
    f"Mean actual PHIF: "
    f"{df['ACTUAL_PHIF_MEAN'].mean():.4f}"
)

print(
    f"Mean interval MAE: "
    f"{df['INTERVAL_MAE'].mean():.4f}"
)

print(
    f"Actual high-PHIF rate "
    f"(mean PHIF >= 0.20): "
    f"{df['ACTUAL_HIGH_PHIF'].mean() * 100:.2f}%"
)


# ============================================================
# TEST SCREENING RULES
# ============================================================

results = []

thickness_values = [
    0.5,
    1.0,
    2.0,
]

high_pct_values = [
    60,
    80,
    90,
]

predicted_mean_values = [
    0.20,
    0.22,
]

for thickness_min in thickness_values:

    for high_pct_min in high_pct_values:

        for predicted_mean_min in predicted_mean_values:

            metrics = evaluate_rule(
                df,
                thickness_min=thickness_min,
                high_pct_min=high_pct_min,
                predicted_mean_min=predicted_mean_min,
            )

            metrics.update(
                {
                    "THICKNESS_MIN": thickness_min,
                    "HIGH_SAMPLE_PCT_MIN": high_pct_min,
                    "PREDICTED_PHIF_MIN": predicted_mean_min,
                    "MAE_MAX": np.nan,
                }
            )

            results.append(metrics)


# ============================================================
# ADD RETROSPECTIVE QUALITY FILTERS
# ============================================================

for thickness_min in [
    0.5,
    1.0,
    2.0,
]:

    for high_pct_min in [
        60,
        80,
        90,
    ]:

        for predicted_mean_min in [
            0.20,
            0.22,
        ]:

            for mae_max in [
                0.05,
                0.08,
            ]:

                metrics = evaluate_rule(
                    df,
                    thickness_min=thickness_min,
                    high_pct_min=high_pct_min,
                    predicted_mean_min=predicted_mean_min,
                    mae_max=mae_max,
                )

                metrics.update(
                    {
                        "THICKNESS_MIN": thickness_min,
                        "HIGH_SAMPLE_PCT_MIN": high_pct_min,
                        "PREDICTED_PHIF_MIN": predicted_mean_min,
                        "MAE_MAX": mae_max,
                    }
                )

                results.append(metrics)


results_df = pd.DataFrame(results)


# ============================================================
# PRINT TOP RULES
# ============================================================

print("\n--- RULES WITHOUT RETROSPECTIVE MAE FILTER ---")

no_mae = results_df[
    results_df["MAE_MAX"].isna()
].copy()

no_mae = no_mae.sort_values(
    [
        "ACTUAL_HIGH_RATE",
        "MEAN_INTERVAL_MAE",
    ],
    ascending=[
        False,
        True,
    ],
)

print(
    no_mae[
        [
            "THICKNESS_MIN",
            "HIGH_SAMPLE_PCT_MIN",
            "PREDICTED_PHIF_MIN",
            "N_INTERVALS",
            "PCT_ALL_INTERVALS",
            "N_WELLS",
            "MEAN_THICKNESS",
            "MEAN_PREDICTED_PHIF",
            "MEAN_ACTUAL_PHIF",
            "MEAN_INTERVAL_MAE",
            "ACTUAL_HIGH_RATE",
            "MEAN_ABS_ERROR",
            "MEAN_BIAS",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# MAE FILTER RESULTS
# ============================================================

print("\n--- RULES WITH RETROSPECTIVE MAE FILTER ---")

with_mae = results_df[
    results_df["MAE_MAX"].notna()
].copy()

with_mae = with_mae.sort_values(
    [
        "ACTUAL_HIGH_RATE",
        "MEAN_INTERVAL_MAE",
    ],
    ascending=[
        False,
        True,
    ],
)

print(
    with_mae[
        [
            "THICKNESS_MIN",
            "HIGH_SAMPLE_PCT_MIN",
            "PREDICTED_PHIF_MIN",
            "MAE_MAX",
            "N_INTERVALS",
            "PCT_ALL_INTERVALS",
            "N_WELLS",
            "MEAN_THICKNESS",
            "MEAN_PREDICTED_PHIF",
            "MEAN_ACTUAL_PHIF",
            "MEAN_INTERVAL_MAE",
            "ACTUAL_HIGH_RATE",
            "MEAN_ABS_ERROR",
            "MEAN_BIAS",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# WELL-LEVEL VALIDATION OF SIMPLE CANDIDATE RULE
# ============================================================

candidate_mask = (
    (df["THICKNESS"] >= 1.0)
    & (df["HIGH_SAMPLE_PCT"] >= 80)
    & (df["PREDICTED_PHIF_MEAN"] >= 0.20)
)

candidate = df.loc[
    candidate_mask
].copy()

print("\n--- CANDIDATE RULE ---")

print(
    "Thickness >= 1.0 m"
)

print(
    "High sample percentage >= 80%"
)

print(
    "Predicted PHIF mean >= 0.20"
)

print(
    f"\nSelected intervals: "
    f"{len(candidate):,}"
)

print(
    f"Selected wells: "
    f"{candidate['WELL'].nunique():,}"
)

if len(candidate) > 0:

    print(
        f"Mean predicted PHIF: "
        f"{candidate['PREDICTED_PHIF_MEAN'].mean():.4f}"
    )

    print(
        f"Mean actual PHIF: "
        f"{candidate['ACTUAL_PHIF_MEAN'].mean():.4f}"
    )

    print(
        f"Actual high-PHIF rate: "
        f"{candidate['ACTUAL_HIGH_PHIF'].mean() * 100:.2f}%"
    )

    print(
        f"Mean interval MAE: "
        f"{candidate['INTERVAL_MAE'].mean():.4f}"
    )

    print(
        f"Mean thickness: "
        f"{candidate['THICKNESS'].mean():.2f} m"
    )

    print(
        "\nCandidate intervals by well:"
    )

    print(
        candidate["WELL"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)