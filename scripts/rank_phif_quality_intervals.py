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
    / "phif_priority_interval_ranking.csv"
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
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# OPERATIONAL SCREENING
# ============================================================

# Tier A:
# Broad candidate interval.
df["SCREENING_TIER"] = np.select(
    [
        (
            (df["THICKNESS"] >= 1.0)
            & (df["HIGH_SAMPLE_PCT"] >= 80)
            & (df["PREDICTED_PHIF_MEAN"] >= 0.20)
        ),
        (
            (df["THICKNESS"] >= 1.0)
            & (df["HIGH_SAMPLE_PCT"] >= 90)
            & (df["PREDICTED_PHIF_MEAN"] >= 0.22)
        ),
    ],
    [
        "A_CANDIDATE",
        "B_STRONG_CANDIDATE",
    ],
    default="BELOW_SCREENING",
)

# Because the strong candidate satisfies the broad
# candidate criteria as well, explicitly assign B first.
strong_mask = (
    (df["THICKNESS"] >= 1.0)
    & (df["HIGH_SAMPLE_PCT"] >= 90)
    & (df["PREDICTED_PHIF_MEAN"] >= 0.22)
)

candidate_mask = (
    (df["THICKNESS"] >= 1.0)
    & (df["HIGH_SAMPLE_PCT"] >= 80)
    & (df["PREDICTED_PHIF_MEAN"] >= 0.20)
)

df.loc[candidate_mask, "SCREENING_TIER"] = "A_CANDIDATE"
df.loc[strong_mask, "SCREENING_TIER"] = "B_STRONG_CANDIDATE"


# ============================================================
# OPERATIONAL SCORE
# ============================================================

# Rank-based normalization is used so that very different
# scales (PHIF, %, meters) can be combined.
#
# IMPORTANT:
# This score uses prediction-side variables only.
# Actual PHIF and interval MAE are NOT included.

def percentile_rank(series):
    return series.rank(
        method="average",
        pct=True,
    )


df["SCORE_PHIF_MEAN"] = percentile_rank(
    df["PREDICTED_PHIF_MEAN"]
)

df["SCORE_HIGH_PCT"] = percentile_rank(
    df["HIGH_SAMPLE_PCT"]
)

df["SCORE_THICKNESS"] = percentile_rank(
    df["THICKNESS"]
)

df["SCORE_PHIF_MAX"] = percentile_rank(
    df["PREDICTED_PHIF_MAX"]
)

df["SCORE_PHIF_MIN"] = percentile_rank(
    df["PREDICTED_PHIF_MIN"]
)


# Weighted score.
#
# Main emphasis:
# - mean predicted PHIF
# - continuity/high-sample percentage
# - thickness
#
# Max/min PHIF receive lower weights.

df["OPERATIONAL_SCORE"] = (
    0.40 * df["SCORE_PHIF_MEAN"]
    + 0.25 * df["SCORE_HIGH_PCT"]
    + 0.20 * df["SCORE_THICKNESS"]
    + 0.10 * df["SCORE_PHIF_MAX"]
    + 0.05 * df["SCORE_PHIF_MIN"]
)


# ============================================================
# RETROSPECTIVE VALIDATION
# ============================================================

df["ACTUAL_HIGH_PHIF"] = (
    df["ACTUAL_PHIF_MEAN"] >= 0.20
)

df["ABS_MEAN_ERROR"] = (
    df["PREDICTED_PHIF_MEAN"]
    - df["ACTUAL_PHIF_MEAN"]
).abs()

df["VALIDATION_STATUS"] = np.select(
    [
        (
            (df["ACTUAL_PHIF_MEAN"] >= 0.20)
            & (df["INTERVAL_MAE"] <= 0.05)
        ),
        (
            (df["ACTUAL_PHIF_MEAN"] >= 0.20)
            | (df["INTERVAL_MAE"] <= 0.05)
        ),
    ],
    [
        "VALIDATED_STRONG",
        "VALIDATED_PARTIAL",
    ],
    default="VALIDATED_WEAK",
)


# ============================================================
# RANK
# ============================================================

df = df.sort_values(
    [
        "SCREENING_TIER",
        "OPERATIONAL_SCORE",
    ],
    ascending=[
        True,
        False,
    ],
)

# Put strong candidates first.
tier_order = {
    "B_STRONG_CANDIDATE": 0,
    "A_CANDIDATE": 1,
    "BELOW_SCREENING": 2,
}

df["_TIER_ORDER"] = (
    df["SCREENING_TIER"]
    .map(tier_order)
)

df = df.sort_values(
    [
        "_TIER_ORDER",
        "OPERATIONAL_SCORE",
    ],
    ascending=[
        True,
        False,
    ],
).reset_index(drop=True)

df["PRIORITY_RANK"] = np.arange(
    1,
    len(df) + 1,
)

df.drop(
    columns=["_TIER_ORDER"],
    inplace=True,
)


# ============================================================
# PRINT OVERALL
# ============================================================

print("=" * 100)
print("PHIF PRIORITY INTERVAL RANKING")
print("=" * 100)

print(
    f"\nTotal intervals: {len(df):,}"
)

print(
    f"Total wells: {df['WELL'].nunique():,}"
)

print("\n--- SCREENING TIER COUNTS ---")

print(
    df["SCREENING_TIER"]
    .value_counts()
    .to_string()
)


# ============================================================
# TOP PRIORITY INTERVALS
# ============================================================

print("\n--- TOP 20 PRIORITY INTERVALS ---")

top_cols = [
    "PRIORITY_RANK",
    "SCREENING_TIER",
    "WELL",
    "TOP_DEPTH",
    "BASE_DEPTH",
    "THICKNESS",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "PREDICTED_PHIF_MAX",
    "PREDICTED_PHIF_MIN",
    "OPERATIONAL_SCORE",
    "ACTUAL_PHIF_MEAN",
    "INTERVAL_MAE",
    "VALIDATION_STATUS",
]

print(
    df[top_cols]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# WELL SUMMARY
# ============================================================

print("\n--- PRIORITY INTERVALS BY WELL ---")

priority = df[
    df["SCREENING_TIER"] != "BELOW_SCREENING"
].copy()

if len(priority) > 0:

    well_summary = (
        priority.groupby("WELL")
        .agg(
            N_PRIORITY_INTERVALS=("WELL", "size"),
            TOTAL_PRIORITY_THICKNESS=("THICKNESS", "sum"),
            MEAN_PREDICTED_PHIF=(
                "PREDICTED_PHIF_MEAN",
                "mean",
            ),
            MAX_PREDICTED_PHIF=(
                "PREDICTED_PHIF_MAX",
                "max",
            ),
            MEAN_OPERATIONAL_SCORE=(
                "OPERATIONAL_SCORE",
                "mean",
            ),
            MEAN_ACTUAL_PHIF=(
                "ACTUAL_PHIF_MEAN",
                "mean",
            ),
            MEAN_INTERVAL_MAE=(
                "INTERVAL_MAE",
                "mean",
            ),
        )
        .sort_values(
            "MEAN_OPERATIONAL_SCORE",
            ascending=False,
        )
    )

    print(
        well_summary.to_string()
    )


# ============================================================
# RETROSPECTIVE PERFORMANCE BY RANK GROUP
# ============================================================

print("\n--- RETROSPECTIVE PERFORMANCE BY RANK GROUP ---")

for n in [5, 10, 20, 30]:

    subset = df.head(n)

    print(
        f"\nTop {n}:"
    )

    print(
        f"Mean predicted PHIF: "
        f"{subset['PREDICTED_PHIF_MEAN'].mean():.4f}"
    )

    print(
        f"Mean actual PHIF: "
        f"{subset['ACTUAL_PHIF_MEAN'].mean():.4f}"
    )

    print(
        f"Actual high-PHIF rate: "
        f"{subset['ACTUAL_HIGH_PHIF'].mean() * 100:.2f}%"
    )

    print(
        f"Mean interval MAE: "
        f"{subset['INTERVAL_MAE'].mean():.4f}"
    )

    print(
        f"Mean thickness: "
        f"{subset['THICKNESS'].mean():.2f} m"
    )


# ============================================================
# SAVE
# ============================================================

output_cols = [
    "PRIORITY_RANK",
    "SCREENING_TIER",
    "WELL",
    "TOP_DEPTH",
    "BASE_DEPTH",
    "THICKNESS",
    "N_HIGH_SAMPLES",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "PREDICTED_PHIF_MAX",
    "PREDICTED_PHIF_MIN",
    "OPERATIONAL_SCORE",
    "ACTUAL_PHIF_MEAN",
    "ACTUAL_PHIF_MAX",
    "INTERVAL_MAE",
    "ABS_MEAN_ERROR",
    "VALIDATION_STATUS",
]

df[output_cols].to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)