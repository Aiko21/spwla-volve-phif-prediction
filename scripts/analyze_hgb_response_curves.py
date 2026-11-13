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
FEATURES = ["GR", "LOG10_RT"]

N_GRID = 50

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": 42,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HGB RESPONSE CURVE ANALYSIS")
print("=" * 80)

print(f"\nDataset:")
print(DATASET)

df = pd.read_csv(DATASET)

# Recreate LOG10_RT deterministically
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


required_columns = [
    "WELL",
    TARGET,
    "GR",
    "LOG10_RT",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


df = df.dropna(
    subset=required_columns
).copy()


wells = sorted(df["WELL"].unique())

print(f"\nDataset shape:")
print(df.shape)

print("\nWells:")
print(wells)


# ============================================================
# RESPONSE CURVE FUNCTION
# ============================================================

def calculate_response_curve(
    model,
    X_reference,
    feature,
    n_grid=50,
):
    """
    Calculate one-dimensional model response.

    The selected feature is varied across a grid while
    the other feature values are kept at their observed
    reference values.

    This is a partial-dependence-style response analysis.
    """

    values = X_reference[feature]

    grid = np.linspace(
        values.quantile(0.02),
        values.quantile(0.98),
        n_grid,
    )

    X_response = X_reference.copy()

    predictions = []

    for value in grid:

        X_response[feature] = value

        pred = model.predict(
            X_response
        )

        predictions.append(
            np.mean(pred)
        )

    return pd.DataFrame(
        {
            "FEATURE": feature,
            "FEATURE_VALUE": grid,
            "MEAN_PREDICTED_PHIF": predictions,
        }
    )


# ============================================================
# STRICT LEAVE-ONE-WELL-OUT RESPONSE ANALYSIS
# ============================================================

all_results = []


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

    # --------------------------------------------------------
    # Train HGB without held-out well
    # --------------------------------------------------------

    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Calculate response curves
    # --------------------------------------------------------

    for feature in FEATURES:

        response = calculate_response_curve(
            model=model,
            X_reference=X_test.copy(),
            feature=feature,
            n_grid=N_GRID,
        )

        response["HELD_OUT_WELL"] = held_out_well

        all_results.append(
            response
        )

        print(
            f"{feature}: "
            f"prediction range = "
            f"{response['MEAN_PREDICTED_PHIF'].min():.6f}"
            f" → "
            f"{response['MEAN_PREDICTED_PHIF'].max():.6f}"
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

response_df = pd.concat(
    all_results,
    ignore_index=True,
)


# ============================================================
# RESPONSE SUMMARY
# ============================================================

summary_rows = []

for (
    well,
    feature,
), group in response_df.groupby(
    ["HELD_OUT_WELL", "FEATURE"]
):

    group = group.sort_values(
        "FEATURE_VALUE"
    )

    pred = group[
        "MEAN_PREDICTED_PHIF"
    ].values

    values = group[
        "FEATURE_VALUE"
    ].values

    delta = pred[-1] - pred[0]

    # Approximate monotonic direction
    if delta > 0.005:
        direction = "INCREASING"
    elif delta < -0.005:
        direction = "DECREASING"
    else:
        direction = "FLAT"

    summary_rows.append(
        {
            "HELD_OUT_WELL": well,
            "FEATURE": feature,
            "PRED_MIN": pred.min(),
            "PRED_MAX": pred.max(),
            "PRED_RANGE": pred.max() - pred.min(),
            "LOW_GRID_PRED": pred[0],
            "HIGH_GRID_PRED": pred[-1],
            "DELTA_LOW_TO_HIGH": delta,
            "DIRECTION": direction,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

response_path = (
    OUTPUT_DIR
    / "phif_hgb_response_curves.csv"
)

summary_path = (
    OUTPUT_DIR
    / "phif_hgb_response_summary.csv"
)

response_df.to_csv(
    response_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RESPONSE SUMMARY")
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
    f"Unique wells analyzed : "
    f"{summary_df['HELD_OUT_WELL'].nunique()}"
)

print(
    f"Expected wells        : "
    f"{len(wells)}"
)

print(
    f"Features analyzed     : "
    f"{summary_df['FEATURE'].nunique()}"
)

print(
    f"Expected features     : "
    f"{len(FEATURES)}"
)

print("\nOutputs:")
print(response_path)
print(summary_path)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)