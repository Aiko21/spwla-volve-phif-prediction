from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "models"
)

TARGET = "PHIF"

FEATURES = [
    "GR",
    "LOG10_RT",
]

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}


# Project-defined porosity classes
def classify_phif(value):

    if value < 0.10:
        return "LOW"

    elif value < 0.20:
        return "MODERATE"

    else:
        return "HIGH"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("PHIF RESERVOIR QUALITY ANALYSIS")
print("=" * 80)

print("\nDataset:")
print(DATASET)

df = pd.read_csv(DATASET)


# ============================================================
# DERIVE LOG10_RT
# ============================================================

if "LOG10_RT" not in df.columns:

    if "RT" not in df.columns:
        raise ValueError(
            "Dataset contains neither LOG10_RT nor RT."
        )

    df["LOG10_RT"] = np.where(
        df["RT"] > 0,
        np.log10(df["RT"]),
        np.nan,
    )

    print(
        "\n[INFO] LOG10_RT was derived from RT using log10(RT)."
    )


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "WELL",
    "DEPTH",
    TARGET,
    "GR",
    "LOG10_RT",
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


wells = sorted(
    df["WELL"].unique()
)

print("\nDataset shape:")
print(df.shape)

print("\nWells:")
print(wells)


# ============================================================
# STRICT LEAVE-ONE-WELL-OUT PREDICTION
# ============================================================

prediction_rows = []


for held_out_well in wells:

    print("\n" + "-" * 80)
    print(
        f"HELD-OUT WELL: {held_out_well}"
    )
    print("-" * 80)

    train_df = df[
        df["WELL"] != held_out_well
    ].copy()

    test_df = df[
        df["WELL"] == held_out_well
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]

    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    fold_predictions = test_df[
        [
            "WELL",
            "DEPTH",
            TARGET,
            "GR",
            "RT",
            "LOG10_RT",
        ]
    ].copy()

    fold_predictions[
        "PREDICTED_PHIF"
    ] = predictions

    fold_predictions[
        "ABS_ERROR"
    ] = np.abs(
        fold_predictions[TARGET]
        - fold_predictions["PREDICTED_PHIF"]
    )

    fold_predictions[
        "ACTUAL_QUALITY"
    ] = fold_predictions[
        TARGET
    ].apply(classify_phif)

    fold_predictions[
        "PREDICTED_QUALITY"
    ] = fold_predictions[
        "PREDICTED_PHIF"
    ].apply(classify_phif)

    fold_predictions[
        "QUALITY_MATCH"
    ] = (
        fold_predictions["ACTUAL_QUALITY"]
        == fold_predictions["PREDICTED_QUALITY"]
    )

    prediction_rows.append(
        fold_predictions
    )

    print(
        f"Rows: {len(fold_predictions):,}"
    )

    print(
        "Predicted PHIF:",
        f"{predictions.min():.6f}",
        "→",
        f"{predictions.max():.6f}",
    )


# ============================================================
# COMBINE
# ============================================================

pred_df = pd.concat(
    prediction_rows,
    ignore_index=True,
)


# ============================================================
# WELL-LEVEL SUMMARY
# ============================================================

well_summary = (
    pred_df
    .groupby("WELL")
    .agg(
        N=("WELL", "size"),

        ACTUAL_PHIF_MEAN=(
            TARGET,
            "mean",
        ),

        ACTUAL_PHIF_MEDIAN=(
            TARGET,
            "median",
        ),

        PREDICTED_PHIF_MEAN=(
            "PREDICTED_PHIF",
            "mean",
        ),

        PREDICTED_PHIF_MEDIAN=(
            "PREDICTED_PHIF",
            "median",
        ),

        MAE=(
            "ABS_ERROR",
            "mean",
        ),

        QUALITY_MATCH_RATE=(
            "QUALITY_MATCH",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# QUALITY DISTRIBUTION
# ============================================================

actual_quality = (
    pd.crosstab(
        pred_df["WELL"],
        pred_df["ACTUAL_QUALITY"],
        normalize="index",
    )
    .fillna(0)
    * 100
)

predicted_quality = (
    pd.crosstab(
        pred_df["WELL"],
        pred_df["PREDICTED_QUALITY"],
        normalize="index",
    )
    .fillna(0)
    * 100
)


# Ensure all expected columns exist
for column in [
    "LOW",
    "MODERATE",
    "HIGH",
]:

    if column not in actual_quality.columns:
        actual_quality[column] = 0

    if column not in predicted_quality.columns:
        predicted_quality[column] = 0


actual_quality = actual_quality[
    [
        "LOW",
        "MODERATE",
        "HIGH",
    ]
].reset_index()


predicted_quality = predicted_quality[
    [
        "LOW",
        "MODERATE",
        "HIGH",
    ]
].reset_index()


actual_quality = actual_quality.rename(
    columns={
        "LOW": "ACTUAL_LOW_PCT",
        "MODERATE": "ACTUAL_MODERATE_PCT",
        "HIGH": "ACTUAL_HIGH_PCT",
    }
)

predicted_quality = predicted_quality.rename(
    columns={
        "LOW": "PREDICTED_LOW_PCT",
        "MODERATE": "PREDICTED_MODERATE_PCT",
        "HIGH": "PREDICTED_HIGH_PCT",
    }
)


well_summary = (
    well_summary
    .merge(
        actual_quality,
        on="WELL",
        how="left",
    )
    .merge(
        predicted_quality,
        on="WELL",
        how="left",
    )
)


# ============================================================
# OVERALL QUALITY DISTRIBUTION
# ============================================================

overall_actual = (
    pred_df["ACTUAL_QUALITY"]
    .value_counts(
        normalize=True
    )
    .mul(100)
)

overall_predicted = (
    pred_df["PREDICTED_QUALITY"]
    .value_counts(
        normalize=True
    )
    .mul(100)
)


# ============================================================
# DEPTH ZONE ANALYSIS
# ============================================================

zone_rows = []

for well, group in pred_df.groupby(
    "WELL"
):

    group = group.sort_values(
        "DEPTH"
    ).copy()

    group[
        "DEPTH_BIN"
    ] = pd.cut(
        group["DEPTH"],
        bins=10,
        duplicates="drop",
    )

    zone_summary = (
        group
        .groupby(
            "DEPTH_BIN",
            observed=True,
        )
        .agg(
            N=("DEPTH", "size"),
            DEPTH_MIN=("DEPTH", "min"),
            DEPTH_MAX=("DEPTH", "max"),
            ACTUAL_PHIF_MEAN=(
                TARGET,
                "mean",
            ),
            PREDICTED_PHIF_MEAN=(
                "PREDICTED_PHIF",
                "mean",
            ),
            PREDICTED_PHIF_MAX=(
                "PREDICTED_PHIF",
                "max",
            ),
            PREDICTED_HIGH_PCT=(
                "PREDICTED_QUALITY",
                lambda x: (
                    x.eq("HIGH").mean()
                    * 100
                ),
            ),
        )
        .reset_index(drop=True)
    )

    zone_summary[
        "WELL"
    ] = well

    zone_rows.append(
        zone_summary
    )


depth_zone_summary = pd.concat(
    zone_rows,
    ignore_index=True,
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_path = (
    OUTPUT_DIR
    / "phif_hgb_reservoir_quality_predictions.csv"
)

well_summary_path = (
    OUTPUT_DIR
    / "phif_hgb_reservoir_quality_by_well.csv"
)

depth_zone_path = (
    OUTPUT_DIR
    / "phif_hgb_reservoir_quality_depth_zones.csv"
)


pred_df.to_csv(
    prediction_path,
    index=False,
)

well_summary.to_csv(
    well_summary_path,
    index=False,
)

depth_zone_summary.to_csv(
    depth_zone_path,
    index=False,
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OVERALL ACTUAL QUALITY DISTRIBUTION")
print("=" * 80)

for quality in [
    "LOW",
    "MODERATE",
    "HIGH",
]:

    print(
        f"{quality:10s}: "
        f"{overall_actual.get(quality, 0):.2f}%"
    )


print("\n" + "=" * 80)
print("OVERALL PREDICTED QUALITY DISTRIBUTION")
print("=" * 80)

for quality in [
    "LOW",
    "MODERATE",
    "HIGH",
]:

    print(
        f"{quality:10s}: "
        f"{overall_predicted.get(quality, 0):.2f}%"
    )


print("\n" + "=" * 80)
print("WELL-LEVEL RESERVOIR QUALITY SUMMARY")
print("=" * 80)

display_columns = [
    "WELL",
    "N",
    "ACTUAL_PHIF_MEAN",
    "PREDICTED_PHIF_MEAN",
    "MAE",
    "QUALITY_MATCH_RATE",
    "PREDICTED_LOW_PCT",
    "PREDICTED_MODERATE_PCT",
    "PREDICTED_HIGH_PCT",
]

print(
    well_summary[
        display_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

print(
    f"Unique wells analyzed : "
    f"{pred_df['WELL'].nunique()}"
)

print(
    f"Expected wells        : "
    f"{len(wells)}"
)

print(
    f"Total prediction rows : "
    f"{len(pred_df):,}"
)

print(
    f"Quality classes       : "
    f"{pred_df['PREDICTED_QUALITY'].nunique()}"
)

print("\nOutputs:")
print(prediction_path)
print(well_summary_path)
print(depth_zone_path)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)