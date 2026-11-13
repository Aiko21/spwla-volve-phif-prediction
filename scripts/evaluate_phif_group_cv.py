from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "phif_ml_dataset_v1.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURES = [
    "GR",
    "RT",
    "LOG10_RT",
]

TARGET = "PHIF"
WELL_COLUMN = "WELL"

# Same XGBoost configuration used in the previous baseline
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHIF GROUP CROSS-VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

# LOG10_RT is a derived feature and is not stored
# in the base ML dataset.
# Create it here to keep the source dataset unchanged.
if "LOG10_RT" not in df.columns:
    if "RT" not in df.columns:
        raise ValueError(
            "RT column is required to create LOG10_RT."
        )

    if (df["RT"] <= 0).any():
        raise ValueError(
            "RT contains non-positive values; "
            "cannot safely compute LOG10_RT."
        )

    df["LOG10_RT"] = np.log10(df["RT"])

    print("Created derived feature: LOG10_RT")

required_columns = FEATURES + [TARGET, WELL_COLUMN]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# VALIDATION
# ============================================================

df = df.dropna(subset=required_columns).copy()

if df.empty:
    raise ValueError("No valid rows available after dropping missing values.")


print(f"Rows used: {len(df):,}")
print(f"Wells: {df[WELL_COLUMN].nunique()}")

wells = sorted(df[WELL_COLUMN].unique())

print("\nWells:")
for well in wells:
    print(f"  - {well}")


# ============================================================
# MODEL FACTORY
# ============================================================

def create_linear_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def create_xgb_model():
    return XGBRegressor(**XGB_PARAMS)


# ============================================================
# GROUP CV
# ============================================================

results = []

predictions = []

print("\n" + "=" * 70)
print("STARTING LEAVE-ONE-WELL-OUT CV")
print("=" * 70)

for fold_number, test_well in enumerate(wells, start=1):

    print("\n" + "-" * 70)
    print(f"Fold {fold_number}/{len(wells)}")
    print(f"Held-out well: {test_well}")
    print("-" * 70)

    train_df = df[df[WELL_COLUMN] != test_well].copy()
    test_df = df[df[WELL_COLUMN] == test_well].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")

    print(
        f"Actual PHIF mean: {y_test.mean():.6f} | "
        f"median: {y_test.median():.6f}"
    )

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    linear_model = create_linear_model()

    linear_model.fit(X_train, y_train)

    linear_pred = linear_model.predict(X_test)

    linear_mae = mean_absolute_error(
        y_test,
        linear_pred
    )

    linear_rmse = np.sqrt(
        mean_squared_error(y_test, linear_pred)
    )

    linear_r2 = r2_score(
        y_test,
        linear_pred
    )

    print("\nLinear Regression:")
    print(f"  MAE :  {linear_mae:.6f}")
    print(f"  RMSE:  {linear_rmse:.6f}")
    print(f"  RÂ²  :  {linear_r2:.6f}")

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_model = create_xgb_model()

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb_model.predict(X_test)

    xgb_mae = mean_absolute_error(
        y_test,
        xgb_pred
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(y_test, xgb_pred)
    )

    xgb_r2 = r2_score(
        y_test,
        xgb_pred
    )

    print("\nXGBoost:")
    print(f"  MAE :  {xgb_mae:.6f}")
    print(f"  RMSE:  {xgb_rmse:.6f}")
    print(f"  RÂ²  :  {xgb_r2:.6f}")

    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    results.append(
        {
            "FOLD": fold_number,
            "HELD_OUT_WELL": test_well,
            "N_TRAIN": len(train_df),
            "N_TEST": len(test_df),
            "ACTUAL_PHIF_MEAN": y_test.mean(),
            "ACTUAL_PHIF_MEDIAN": y_test.median(),

            "LINEAR_MAE": linear_mae,
            "LINEAR_RMSE": linear_rmse,
            "LINEAR_R2": linear_r2,

            "XGB_MAE": xgb_mae,
            "XGB_RMSE": xgb_rmse,
            "XGB_R2": xgb_r2,
        }
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    fold_predictions = test_df[
        [WELL_COLUMN, "DEPTH", TARGET]
    ].copy()

    fold_predictions["FOLD"] = fold_number

    fold_predictions["LINEAR_PRED"] = linear_pred
    fold_predictions["LINEAR_RESIDUAL"] = (
        y_test.to_numpy() - linear_pred
    )

    fold_predictions["XGB_PRED"] = xgb_pred
    fold_predictions["XGB_RESIDUAL"] = (
        y_test.to_numpy() - xgb_pred
    )

    predictions.append(fold_predictions)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    predictions,
    ignore_index=True
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary_rows = []

for model_name in ["LINEAR", "XGB"]:

    mae_values = results_df[f"{model_name}_MAE"]
    rmse_values = results_df[f"{model_name}_RMSE"]
    r2_values = results_df[f"{model_name}_R2"]

    summary_rows.append(
        {
            "MODEL": model_name,

            "MEAN_MAE": mae_values.mean(),
            "MEDIAN_MAE": mae_values.median(),
            "STD_MAE": mae_values.std(ddof=1),

            "MEAN_RMSE": rmse_values.mean(),
            "MEDIAN_RMSE": rmse_values.median(),
            "STD_RMSE": rmse_values.std(ddof=1),

            "MEAN_R2": r2_values.mean(),
            "MEDIAN_R2": r2_values.median(),
            "STD_R2": r2_values.std(ddof=1),

            "MIN_R2": r2_values.min(),
            "MAX_R2": r2_values.max(),

            "N_WELLS": len(results_df),
        }
    )

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    OUTPUT_DIR /
    "phif_group_cv_results.csv"
)

summary_path = (
    OUTPUT_DIR /
    "phif_group_cv_summary.csv"
)

predictions_path = (
    OUTPUT_DIR /
    "phif_group_cv_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("GROUP CV RESULTS")
print("=" * 70)

print("\nPer-well results:")
print(
    results_df[
        [
            "HELD_OUT_WELL",
            "LINEAR_MAE",
            "LINEAR_RMSE",
            "LINEAR_R2",
            "XGB_MAE",
            "XGB_RMSE",
            "XGB_R2",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\n{results_path}")
print(summary_path)
print(predictions_path)

print("\nDone.")
