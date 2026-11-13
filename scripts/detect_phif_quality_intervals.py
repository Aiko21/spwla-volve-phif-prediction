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
    / "phif_hgb_reservoir_quality_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
)

OUTPUT_INTERVALS = (
    OUTPUT_DIR
    / "phif_hgb_high_quality_intervals.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "phif_hgb_high_quality_interval_summary.csv"
)


# Project-defined high-porosity threshold
HIGH_PHIF_THRESHOLD = 0.20

# Minimum number of consecutive samples
MIN_CONSECUTIVE_SAMPLES = 5

# Maximum allowed depth gap when merging adjacent samples
MAX_DEPTH_GAP = 1.0


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 80)
print("PHIF HIGH-QUALITY INTERVAL DETECTION")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "WELL",
    "DEPTH",
    "PHIF",
    "PREDICTED_PHIF",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df.dropna(
    subset=required_columns
).copy()

df = df.sort_values(
    [
        "WELL",
        "DEPTH",
    ]
).reset_index(drop=True)


print("\nDataset shape:")
print(df.shape)

print("\nWells:")
print(
    sorted(
        df["WELL"].unique()
    )
)


# ============================================================
# IDENTIFY HIGH-POROSITY SAMPLES
# ============================================================

df["IS_HIGH"] = (
    df["PREDICTED_PHIF"]
    >= HIGH_PHIF_THRESHOLD
)


# ============================================================
# INTERVAL DETECTION
# ============================================================

interval_rows = []


for well, group in df.groupby(
    "WELL",
    sort=True,
):

    group = (
        group
        .sort_values("DEPTH")
        .reset_index(drop=True)
    )

    high = group[
        group["IS_HIGH"]
    ].copy()

    if high.empty:
        print(
            f"\n{well}: no high-porosity samples."
        )
        continue

    # --------------------------------------------------------
    # Identify breaks between high samples
    # --------------------------------------------------------

    depth_gap = (
        high["DEPTH"]
        .diff()
        .fillna(0)
    )

    new_interval = (
        depth_gap > MAX_DEPTH_GAP
    )

    high["INTERVAL_ID"] = (
        new_interval
        .cumsum()
    )

    # --------------------------------------------------------
    # Summarize candidate intervals
    # --------------------------------------------------------

    for interval_id, interval in high.groupby(
        "INTERVAL_ID"
    ):

        n_samples = len(interval)

        if n_samples < MIN_CONSECUTIVE_SAMPLES:
            continue

        top_depth = (
            interval["DEPTH"].min()
        )

        base_depth = (
            interval["DEPTH"].max()
        )

        thickness = (
            base_depth
            - top_depth
        )

        predicted_mean = (
            interval["PREDICTED_PHIF"].mean()
        )

        predicted_max = (
            interval["PREDICTED_PHIF"].max()
        )

        predicted_min = (
            interval["PREDICTED_PHIF"].min()
        )

        actual_mean = (
            interval["PHIF"].mean()
        )

        actual_max = (
            interval["PHIF"].max()
        )

        interval_mae = (
            np.abs(
                interval["PHIF"]
                - interval["PREDICTED_PHIF"]
            ).mean()
        )

        # Percentage of all samples in the physical
        # interval that satisfy the high threshold.
        full_interval = group[
            (group["DEPTH"] >= top_depth)
            & (group["DEPTH"] <= base_depth)
        ]

        high_percentage = (
            full_interval["IS_HIGH"].mean()
            * 100
        )

        interval_rows.append(
            {
                "WELL": well,
                "TOP_DEPTH": top_depth,
                "BASE_DEPTH": base_depth,
                "THICKNESS": thickness,
                "N_HIGH_SAMPLES": n_samples,
                "HIGH_SAMPLE_PCT": high_percentage,
                "PREDICTED_PHIF_MEAN": predicted_mean,
                "PREDICTED_PHIF_MAX": predicted_max,
                "PREDICTED_PHIF_MIN": predicted_min,
                "ACTUAL_PHIF_MEAN": actual_mean,
                "ACTUAL_PHIF_MAX": actual_max,
                "INTERVAL_MAE": interval_mae,
            }
        )


# ============================================================
# CREATE OUTPUT
# ============================================================

interval_df = pd.DataFrame(
    interval_rows
)


if interval_df.empty:

    print(
        "\nNo qualifying high-porosity intervals "
        "were detected."
    )

    raise SystemExit(0)


interval_df = (
    interval_df
    .sort_values(
        [
            "WELL",
            "TOP_DEPTH",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# RANK INTERVALS
# ============================================================

interval_df[
    "QUALITY_SCORE"
] = (
    interval_df[
        "PREDICTED_PHIF_MEAN"
    ]
    * interval_df[
        "HIGH_SAMPLE_PCT"
    ]
    / 100
)


interval_df[
    "RANK_WITHIN_WELL"
] = (
    interval_df
    .groupby("WELL")[
        "QUALITY_SCORE"
    ]
    .rank(
        method="first",
        ascending=False,
    )
)


# ============================================================
# WELL SUMMARY
# ============================================================

summary_rows = []


for well, group in df.groupby(
    "WELL"
):

    intervals = interval_df[
        interval_df["WELL"] == well
    ]

    predicted_high_pct = (
        group["IS_HIGH"].mean()
        * 100
    )

    if intervals.empty:

        summary_rows.append(
            {
                "WELL": well,
                "N_HIGH_INTERVALS": 0,
                "TOTAL_HIGH_THICKNESS": 0.0,
                "LONGEST_INTERVAL": 0.0,
                "MAX_PREDICTED_PHIF": (
                    group["PREDICTED_PHIF"].max()
                ),
                "MEAN_PREDICTED_PHIF": (
                    group["PREDICTED_PHIF"].mean()
                ),
                "PREDICTED_HIGH_SAMPLE_PCT": (
                    predicted_high_pct
                ),
            }
        )

        continue

    summary_rows.append(
        {
            "WELL": well,
            "N_HIGH_INTERVALS": len(intervals),
            "TOTAL_HIGH_THICKNESS": (
                intervals["THICKNESS"].sum()
            ),
            "LONGEST_INTERVAL": (
                intervals["THICKNESS"].max()
            ),
            "MAX_PREDICTED_PHIF": (
                group["PREDICTED_PHIF"].max()
            ),
            "MEAN_PREDICTED_PHIF": (
                group["PREDICTED_PHIF"].mean()
            ),
            "PREDICTED_HIGH_SAMPLE_PCT": (
                predicted_high_pct
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

interval_df.to_csv(
    OUTPUT_INTERVALS,
    index=False,
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("DETECTION PARAMETERS")
print("=" * 80)

print(
    f"High PHIF threshold : "
    f"{HIGH_PHIF_THRESHOLD}"
)

print(
    f"Minimum samples     : "
    f"{MIN_CONSECUTIVE_SAMPLES}"
)

print(
    f"Maximum depth gap   : "
    f"{MAX_DEPTH_GAP}"
)


print("\n" + "=" * 80)
print("HIGH-POROSITY INTERVALS")
print("=" * 80)

display_columns = [
    "WELL",
    "TOP_DEPTH",
    "BASE_DEPTH",
    "THICKNESS",
    "N_HIGH_SAMPLES",
    "HIGH_SAMPLE_PCT",
    "PREDICTED_PHIF_MEAN",
    "PREDICTED_PHIF_MAX",
    "ACTUAL_PHIF_MEAN",
    "INTERVAL_MAE",
    "RANK_WITHIN_WELL",
]

print(
    interval_df[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("WELL SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(
    f"Input rows           : "
    f"{len(df):,}"
)

print(
    f"Detected intervals   : "
    f"{len(interval_df):,}"
)

print(
    f"Wells with intervals : "
    f"{interval_df['WELL'].nunique()}"
)

print(
    f"Expected wells       : "
    f"{df['WELL'].nunique()}"
)

print("\nOutputs:")
print(OUTPUT_INTERVALS)
print(OUTPUT_SUMMARY)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)