from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

SPLIT_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "phif_well_split_metadata.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "reports"
    / "models"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42


# ============================================================
# XGBOOST BASELINE PARAMETERS
# ============================================================

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

print("=" * 80)
print("XGBOOST PHIF BASELINE")
print("=" * 80)

print("\nLoading dataset:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "WELL",
    "GR",
    "RT",
    "PHIF",
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


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["LOG10_RT"] = np.log10(
    df["RT"].clip(lower=1e-6)
)

FEATURES = [
    "GR",
    "RT",
    "LOG10_RT",
]

TARGET = "PHIF"


# ============================================================
# LOAD WELL SPLIT
# ============================================================

print("\nLoading well split:")
print(SPLIT_PATH)

split_df = pd.read_csv(SPLIT_PATH)

print(split_df.to_string(index=False))


# ============================================================
# IDENTIFY SPLIT COLUMN
# ============================================================

possible_split_columns = [
    "SPLIT",
    "SET",
    "DATASET",
    "PARTITION",
]

split_column = None

for column in possible_split_columns:

    if column in split_df.columns:
        split_column = column
        break

if split_column is None:

    raise ValueError(
        "Could not identify split column. "
        f"Available columns: {list(split_df.columns)}"
    )


# ============================================================
# IDENTIFY WELL COLUMN
# ============================================================

possible_well_columns = [
    "WELL",
    "Well",
    "well",
]

well_column = None

for column in possible_well_columns:

    if column in split_df.columns:
        well_column = column
        break

if well_column is None:

    raise ValueError(
        "Could not identify WELL column in split metadata."
    )


# ============================================================
# NORMALIZE SPLIT LABELS
# ============================================================

split_df[split_column] = (
    split_df[split_column]
    .astype(str)
    .str.upper()
    .str.strip()
)

train_wells = split_df.loc[
    split_df[split_column] == "TRAIN",
    well_column,
].tolist()

validation_wells = split_df.loc[
    split_df[split_column].isin(
        ["VALIDATION", "VALID", "VAL"]
    ),
    well_column,
].tolist()

test_wells = split_df.loc[
    split_df[split_column] == "TEST",
    well_column,
].tolist()


print("\n" + "=" * 80)
print("WELL SPLIT")
print("=" * 80)

print("\nTRAIN:")
print(train_wells)

print("\nVALIDATION:")
print(validation_wells)

print("\nTEST:")
print(test_wells)


# ============================================================
# CREATE SPLITS
# ============================================================

train_df = df[
    df["WELL"].isin(train_wells)
].copy()

validation_df = df[
    df["WELL"].isin(validation_wells)
].copy()

test_df = df[
    df["WELL"].isin(test_wells)
].copy()


# ============================================================
# VALIDATE SPLIT
# ============================================================

if set(train_df["WELL"]) & set(validation_df["WELL"]):
    raise ValueError("Train/validation well overlap detected.")

if set(train_df["WELL"]) & set(test_df["WELL"]):
    raise ValueError("Train/test well overlap detected.")

if set(validation_df["WELL"]) & set(test_df["WELL"]):
    raise ValueError("Validation/test well overlap detected.")


print("\n" + "=" * 80)
print("SPLIT ROW COUNTS")
print("=" * 80)

print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")
print(f"Total:      {len(train_df) + len(validation_df) + len(test_df):,}")


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)

print("\nParameters:")

for key, value in XGB_PARAMS.items():
    print(f"{key}: {value}")


model = XGBRegressor(
    **XGB_PARAMS
)

model.fit(
    X_train,
    y_train,
    eval_set=[
        (X_val, y_val)
    ],
    verbose=False,
)


# ============================================================
# VALIDATION EVALUATION
# ============================================================

val_pred = model.predict(X_val)

validation_mae = mean_absolute_error(
    y_val,
    val_pred,
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred,
    )
)

validation_r2 = r2_score(
    y_val,
    val_pred,
)


print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

print(f"MAE:  {validation_mae:.6f}")
print(f"RMSE: {validation_rmse:.6f}")
print(f"RÂ²:   {validation_r2:.6f}")


# ============================================================
# REFIT ON TRAIN + VALIDATION
# ============================================================

train_val_df = pd.concat(
    [
        train_df,
        validation_df,
    ],
    ignore_index=True,
)

X_train_val = train_val_df[FEATURES]
y_train_val = train_val_df[TARGET]


print("\n" + "=" * 80)
print("REFITTING ON TRAIN + VALIDATION")
print("=" * 80)

final_model = XGBRegressor(
    **XGB_PARAMS
)

final_model.fit(
    X_train_val,
    y_train_val,
    verbose=False,
)


# ============================================================
# TEST EVALUATION
# ============================================================

test_pred = final_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_pred,
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred,
    )
)

test_r2 = r2_score(
    y_test,
    test_pred,
)


print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(f"MAE:  {test_mae:.6f}")
print(f"RMSE: {test_rmse:.6f}")
print(f"RÂ²:   {test_r2:.6f}")


# ============================================================
# TEST PREDICTIONS
# ============================================================

predictions = test_df[
    ["WELL"]
].copy()

predictions["PHIF"] = y_test.values
predictions["PHIF_PRED"] = test_pred

predictions["RESIDUAL"] = (
    predictions["PHIF"]
    - predictions["PHIF_PRED"]
)

predictions["ABS_ERROR"] = (
    predictions["RESIDUAL"]
    .abs()
)

prediction_path = (
    OUTPUT_DIR /
    "phif_xgboost_test_predictions.csv"
)

predictions.to_csv(
    prediction_path,
    index=False,
)


# ============================================================
# PER-WELL PERFORMANCE
# ============================================================

per_well_records = []

for well, group in predictions.groupby("WELL"):

    actual = group["PHIF"]
    predicted = group["PHIF_PRED"]

    per_well_records.append(
        {
            "WELL": well,
            "ROWS": len(group),
            "MAE": mean_absolute_error(
                actual,
                predicted,
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),
            "R2": r2_score(
                actual,
                predicted,
            ),
            "ACTUAL_MEAN": actual.mean(),
            "PRED_MEAN": predicted.mean(),
            "ACTUAL_STD": actual.std(),
            "PRED_STD": predicted.std(),
        }
    )


per_well_df = pd.DataFrame(
    per_well_records
)

per_well_path = (
    OUTPUT_DIR /
    "phif_xgboost_test_per_well.csv"
)

per_well_df.to_csv(
    per_well_path,
    index=False,
)


print("\n" + "=" * 80)
print("PER-WELL TEST RESULTS")
print("=" * 80)

print(
    per_well_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "FEATURE": FEATURES,
        "IMPORTANCE": final_model.feature_importances_,
    }
).sort_values(
    "IMPORTANCE",
    ascending=False,
)

importance_path = (
    OUTPUT_DIR /
    "phif_xgboost_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False,
)


print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

print(
    importance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# MODEL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "MODEL": "XGBoost_Baseline",
            "FEATURES": ",".join(FEATURES),
            "VALIDATION_MAE": validation_mae,
            "VALIDATION_RMSE": validation_rmse,
            "VALIDATION_R2": validation_r2,
            "TEST_MAE": test_mae,
            "TEST_RMSE": test_rmse,
            "TEST_R2": test_r2,
            "RANDOM_STATE": RANDOM_STATE,
        }
    ]
)

summary_path = (
    OUTPUT_DIR /
    "phif_xgboost_baseline_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST BASELINE COMPLETE")
print("=" * 80)

print("\nSaved:")
print(prediction_path)
print(per_well_path)
print(importance_path)
print(summary_path)
