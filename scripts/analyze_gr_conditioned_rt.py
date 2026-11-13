from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_BINNED_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "gr_conditioned_rt_phif.csv"
)

OUTPUT_SUMMARY_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "gr_conditioned_rt_phif_summary.csv"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_cols = ["WELL", "GR", "RT", "PHIF"]

missing = [
    col for col in required_cols
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


df = df[required_cols].copy()

df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

df = df.dropna(
    subset=required_cols
)

df = df[df["RT"] > 0].copy()

df["LOG10_RT"] = np.log10(df["RT"])


# ============================================================
# GR REGIMES
# ============================================================
#
# IMPORTANT:
# GR bins are calculated WITHIN each well.
#
# This asks:
# "Within a well, does the RT -> PHIF relationship
# change between low-, medium-, and high-GR intervals?"
#
# We use terciles:
#   LOW_GR
#   MID_GR
#   HIGH_GR
# ============================================================

results = []

for well, well_df in df.groupby("WELL", sort=True):

    well_df = well_df.copy()

    try:
        well_df["GR_REGIME"] = pd.qcut(
            well_df["GR"],
            q=3,
            labels=[
                "LOW_GR",
                "MID_GR",
                "HIGH_GR",
            ],
            duplicates="drop",
        )
    except ValueError:
        print(
            f"Skipping {well}: insufficient unique GR values."
        )
        continue

    # ========================================================
    # RT BINS WITHIN EACH WELL + GR REGIME
    # ========================================================

    for regime, regime_df in well_df.groupby(
        "GR_REGIME",
        observed=False,
    ):

        regime_df = regime_df.copy()

        try:
            regime_df["RT_BIN"] = pd.qcut(
                regime_df["LOG10_RT"],
                q=4,
                labels=[
                    "Q1",
                    "Q2",
                    "Q3",
                    "Q4",
                ],
                duplicates="drop",
            )
        except ValueError:
            continue

        grouped = (
            regime_df
            .groupby(
                "RT_BIN",
                observed=False,
            )
            .agg(
                N=("PHIF", "size"),
                GR_MEAN=("GR", "mean"),
                GR_MEDIAN=("GR", "median"),
                RT_MEDIAN=("RT", "median"),
                LOG10_RT_MEDIAN=("LOG10_RT", "median"),
                PHIF_MEAN=("PHIF", "mean"),
                PHIF_MEDIAN=("PHIF", "median"),
                PHIF_STD=("PHIF", "std"),
            )
            .reset_index()
        )

        grouped["WELL"] = well
        grouped["GR_REGIME"] = regime

        results.append(grouped)


# ============================================================
# COMBINE
# ============================================================

binned = pd.concat(
    results,
    ignore_index=True,
)


binned = binned[
    [
        "WELL",
        "GR_REGIME",
        "RT_BIN",
        "N",
        "GR_MEAN",
        "GR_MEDIAN",
        "RT_MEDIAN",
        "LOG10_RT_MEDIAN",
        "PHIF_MEAN",
        "PHIF_MEDIAN",
        "PHIF_STD",
    ]
]


# ============================================================
# Q1 -> Q4 EFFECT WITHIN EACH GR REGIME
# ============================================================

summary = []

for (well, regime), group in binned.groupby(
    ["WELL", "GR_REGIME"],
    sort=True,
):

    group = group.sort_values("RT_BIN")

    if len(group) < 4:
        continue

    q1 = group.iloc[0]
    q4 = group.iloc[-1]

    delta_mean = (
        q4["PHIF_MEAN"]
        - q1["PHIF_MEAN"]
    )

    delta_median = (
        q4["PHIF_MEDIAN"]
        - q1["PHIF_MEDIAN"]
    )

    # Binned monotonicity
    group["BIN_ORDER"] = range(
        1,
        len(group) + 1,
    )

    mean_spearman = group[
        ["BIN_ORDER", "PHIF_MEAN"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    median_spearman = group[
        ["BIN_ORDER", "PHIF_MEDIAN"]
    ].corr(
        method="spearman"
    ).iloc[0, 1]

    summary.append(
        {
            "WELL": well,
            "GR_REGIME": regime,
            "N_TOTAL": group["N"].sum(),

            "GR_MEDIAN": group[
                "GR_MEDIAN"
            ].mean(),

            "Q1_PHIF_MEAN": q1[
                "PHIF_MEAN"
            ],

            "Q4_PHIF_MEAN": q4[
                "PHIF_MEAN"
            ],

            "DELTA_PHIF_MEAN_Q1_Q4": delta_mean,

            "Q1_PHIF_MEDIAN": q1[
                "PHIF_MEDIAN"
            ],

            "Q4_PHIF_MEDIAN": q4[
                "PHIF_MEDIAN"
            ],

            "DELTA_PHIF_MEDIAN_Q1_Q4": delta_median,

            "BINNED_PHIF_MEAN_SPEARMAN": (
                mean_spearman
            ),

            "BINNED_PHIF_MEDIAN_SPEARMAN": (
                median_spearman
            ),
        }
    )


summary_df = pd.DataFrame(summary)


# ============================================================
# PRINT DETAILED TABLE
# ============================================================

print("=" * 120)
print("GR-CONDITIONED RT -> PHIF RELATIONSHIP")
print("=" * 120)
print()

print(
    binned[
        [
            "WELL",
            "GR_REGIME",
            "RT_BIN",
            "N",
            "GR_MEDIAN",
            "RT_MEDIAN",
            "PHIF_MEAN",
            "PHIF_MEDIAN",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 120)
print("GR-CONDITIONED RT EFFECT SUMMARY")
print("=" * 120)
print()

print(
    summary_df[
        [
            "WELL",
            "GR_REGIME",
            "N_TOTAL",
            "GR_MEDIAN",
            "DELTA_PHIF_MEAN_Q1_Q4",
            "DELTA_PHIF_MEDIAN_Q1_Q4",
            "BINNED_PHIF_MEAN_SPEARMAN",
            "BINNED_PHIF_MEDIAN_SPEARMAN",
        ]
    ].to_string(index=False)
)


# ============================================================
# PER-WELL REGIME CONTRAST
# ============================================================

print()
print("=" * 120)
print("REGIME CONTRAST BY WELL")
print("=" * 120)
print()

for well, group in summary_df.groupby(
    "WELL",
    sort=True,
):

    print()
    print(f"--- {well} ---")

    print(
        group[
            [
                "GR_REGIME",
                "GR_MEDIAN",
                "DELTA_PHIF_MEAN_Q1_Q4",
                "DELTA_PHIF_MEDIAN_Q1_Q4",
                "BINNED_PHIF_MEAN_SPEARMAN",
            ]
        ].to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

binned.to_csv(
    OUTPUT_BINNED_PATH,
    index=False,
)

summary_df.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False,
)


print()
print("=" * 120)
print("OUTPUT")
print("=" * 120)
print(OUTPUT_BINNED_PATH)
print(OUTPUT_SUMMARY_PATH)
print()
print("Done.")