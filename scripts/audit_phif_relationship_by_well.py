from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "GR",
    "RT",
    "LOG10_RT",
]

TARGET = "PHIF"
WELL_COLUMN = "WELL"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHIF RELATIONSHIP BY WELL AUDIT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# CREATE DERIVED FEATURE
# ============================================================

if "LOG10_RT" not in df.columns:

    if "RT" not in df.columns:
        raise ValueError(
            "RT column is required to create LOG10_RT."
        )

    if (df["RT"] <= 0).any():
        raise ValueError(
            "RT contains non-positive values."
        )

    df["LOG10_RT"] = np.log10(df["RT"])

    print("Created derived feature: LOG10_RT")


# ============================================================
# VALIDATE
# ============================================================

required_columns = [
    WELL_COLUMN,
    "GR",
    "RT",
    "LOG10_RT",
    TARGET,
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


df = df.dropna(
    subset=required_columns
).copy()

print(f"Rows used: {len(df):,}")
print(f"Wells: {df[WELL_COLUMN].nunique()}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def correlation_stats(
    x: pd.Series,
    y: pd.Series,
):
    """
    Calculate Pearson and Spearman correlations.
    """

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    return (
        pearson_r,
        pearson_p,
        spearman_r,
        spearman_p,
    )


def simple_linear_fit(
    x: pd.Series,
    y: pd.Series,
):
    """
    Fit simple linear regression:

        PHIF = intercept + slope * X
    """

    model = LinearRegression()

    X = x.to_numpy().reshape(-1, 1)
    y_array = y.to_numpy()

    model.fit(X, y_array)

    predictions = model.predict(X)

    r2 = model.score(X, y_array)

    return (
        float(model.coef_[0]),
        float(model.intercept_),
        float(r2),
        float(
            np.sqrt(
                np.mean(
                    (y_array - predictions) ** 2
                )
            )
        ),
    )


# ============================================================
# PER-WELL RELATIONSHIP ANALYSIS
# ============================================================

results = []

print("\n" + "=" * 70)
print("PER-WELL RELATIONSHIPS")
print("=" * 70)

for well in sorted(df[WELL_COLUMN].unique()):

    well_df = df[
        df[WELL_COLUMN] == well
    ].copy()

    y = well_df[TARGET]

    # --------------------------------------------------------
    # Basic target statistics
    # --------------------------------------------------------

    phif_mean = y.mean()
    phif_std = y.std()
    phif_median = y.median()

    phif_q10 = y.quantile(0.10)
    phif_q25 = y.quantile(0.25)
    phif_q75 = y.quantile(0.75)
    phif_q90 = y.quantile(0.90)

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    gr_pearson, gr_pearson_p, gr_spearman, gr_spearman_p = (
        correlation_stats(
            well_df["GR"],
            y,
        )
    )

    rt_pearson, rt_pearson_p, rt_spearman, rt_spearman_p = (
        correlation_stats(
            well_df["RT"],
            y,
        )
    )

    logrt_pearson, logrt_pearson_p, logrt_spearman, logrt_spearman_p = (
        correlation_stats(
            well_df["LOG10_RT"],
            y,
        )
    )

    # --------------------------------------------------------
    # Simple linear relationships
    # --------------------------------------------------------

    gr_slope, gr_intercept, gr_r2, gr_rmse = (
        simple_linear_fit(
            well_df["GR"],
            y,
        )
    )

    rt_slope, rt_intercept, rt_r2, rt_rmse = (
        simple_linear_fit(
            well_df["RT"],
            y,
        )
    )

    logrt_slope, logrt_intercept, logrt_r2, logrt_rmse = (
        simple_linear_fit(
            well_df["LOG10_RT"],
            y,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append(
        {
            "WELL": well,
            "N_ROWS": len(well_df),

            "PHIF_MEAN": phif_mean,
            "PHIF_STD": phif_std,
            "PHIF_MEDIAN": phif_median,
            "PHIF_Q10": phif_q10,
            "PHIF_Q25": phif_q25,
            "PHIF_Q75": phif_q75,
            "PHIF_Q90": phif_q90,

            # GR
            "GR_PEARSON_R": gr_pearson,
            "GR_PEARSON_P": gr_pearson_p,
            "GR_SPEARMAN_R": gr_spearman,
            "GR_SPEARMAN_P": gr_spearman_p,
            "GR_SLOPE": gr_slope,
            "GR_INTERCEPT": gr_intercept,
            "GR_SIMPLE_R2": gr_r2,
            "GR_SIMPLE_RMSE": gr_rmse,

            # RT
            "RT_PEARSON_R": rt_pearson,
            "RT_PEARSON_P": rt_pearson_p,
            "RT_SPEARMAN_R": rt_spearman,
            "RT_SPEARMAN_P": rt_spearman_p,
            "RT_SLOPE": rt_slope,
            "RT_INTERCEPT": rt_intercept,
            "RT_SIMPLE_R2": rt_r2,
            "RT_SIMPLE_RMSE": rt_rmse,

            # LOG10 RT
            "LOGRT_PEARSON_R": logrt_pearson,
            "LOGRT_PEARSON_P": logrt_pearson_p,
            "LOGRT_SPEARMAN_R": logrt_spearman,
            "LOGRT_SPEARMAN_P": logrt_spearman_p,
            "LOGRT_SLOPE": logrt_slope,
            "LOGRT_INTERCEPT": logrt_intercept,
            "LOGRT_SIMPLE_R2": logrt_r2,
            "LOGRT_SIMPLE_RMSE": logrt_rmse,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "WELL"
).reset_index(drop=True)


# ============================================================
# SAVE FULL RESULTS
# ============================================================

output_path = (
    OUTPUT_DIR
    / "phif_relationship_by_well.csv"
)

results_df.to_csv(
    output_path,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION BY WELL")
print("=" * 70)

print(
    results_df[
        [
            "WELL",
            "N_ROWS",
            "PHIF_MEAN",
            "PHIF_STD",
            "PHIF_MEDIAN",
            "PHIF_Q10",
            "PHIF_Q25",
            "PHIF_Q75",
            "PHIF_Q90",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("GR -> PHIF")
print("=" * 70)

print(
    results_df[
        [
            "WELL",
            "GR_PEARSON_R",
            "GR_SPEARMAN_R",
            "GR_SLOPE",
            "GR_SIMPLE_R2",
            "GR_SIMPLE_RMSE",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("RT -> PHIF")
print("=" * 70)

print(
    results_df[
        [
            "WELL",
            "RT_PEARSON_R",
            "RT_SPEARMAN_R",
            "RT_SLOPE",
            "RT_SIMPLE_R2",
            "RT_SIMPLE_RMSE",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("LOG10(RT) -> PHIF")
print("=" * 70)

print(
    results_df[
        [
            "WELL",
            "LOGRT_PEARSON_R",
            "LOGRT_SPEARMAN_R",
            "LOGRT_SLOPE",
            "LOGRT_SIMPLE_R2",
            "LOGRT_SIMPLE_RMSE",
        ]
    ].to_string(index=False)
)


# ============================================================
# RANGE OF RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("RELATIONSHIP VARIABILITY")
print("=" * 70)

for column in [
    "GR_SPEARMAN_R",
    "RT_SPEARMAN_R",
    "LOGRT_SPEARMAN_R",
    "GR_SLOPE",
    "RT_SLOPE",
    "LOGRT_SLOPE",
]:

    print(
        f"{column:25s} "
        f"min={results_df[column].min(): .6f} "
        f"max={results_df[column].max(): .6f}"
    )


print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(output_path)

print("\nDone.")