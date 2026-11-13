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

DATA_PATH = (
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PHIF"
WELL_COLUMN = "WELL"

RANDOM_STATE = 42


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "A_GR_ONLY": [
        "GR",
    ],

    "B_GR_RT": [
        "GR",
        "RT",
        "LOG10_RT",
    ],

    "C_GR_RT_DEPTH": [
        "GR",
        "RT",
        "LOG10_RT",
        "DEPTH",
    ],
}


# ============================================================
# XGBOOST CONFIG
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

print("=" * 70)
print("PHIF DEPTH FEATURE GROUP CV")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# DERIVED FEATURE
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
# VALIDATE BASE COLUMNS
# ============================================================

required_columns = {
    TARGET,
    WELL_COLUMN,
    "GR",
    "RT",
    "LOG10_RT",
    "DEPTH",
}

missing_columns = sorted(
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# DROP INVALID ROWS
# ============================================================

df = df.dropna(
    subset=list(required_columns)
).copy()

print(f"Rows used: {len(df):,}")
print(f"Wells: {df[WELL_COLUMN].nunique()}")


# ============================================================
# MODEL FACTORIES
# ============================================================

def create_linear_model():

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def create_xgb_model():

    return XGBRegressor(
        **XGB_PARAMS
    )


# ============================================================
# GROUP CV
# ============================================================

results = []

predictions = []

wells = sorted(
    df[WELL_COLUMN].unique()
)


print("\n" + "=" * 70)
print("STARTING GROUP CV")
print("=" * 70)


for feature_set_name, features in FEATURE_SETS.items():

    print("\n")
    print("#" * 70)
    print(f"FEATURE SET: {feature_set_name}")
    print("#" * 70)

    print(
        "Features:",
        ", ".join(features)
    )

    for fold_number, test_well in enumerate(
        wells,
        start=1,
    ):

        train_df = df[
            df[WELL_COLUMN] != test_well
        ].copy()

        test_df = df[
            df[WELL_COLUMN] == test_well
        ].copy()

        X_train = train_df[features]
        y_train = train_df[TARGET]

        X_test = test_df[features]
        y_test = test_df[TARGET]


        print(
            f"\nFold {fold_number}/{len(wells)}"
            f" | Held-out: {test_well}"
        )

        # ----------------------------------------------------
        # Linear Regression
        # ----------------------------------------------------

        linear_model = create_linear_model()

        linear_model.fit(
            X_train,
            y_train,
        )

        linear_pred = linear_model.predict(
            X_test
        )

        linear_mae = mean_absolute_error(
            y_test,
            linear_pred,
        )

        linear_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                linear_pred,
            )
        )

        linear_r2 = r2_score(
            y_test,
            linear_pred,
        )


        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        xgb_model = create_xgb_model()

        xgb_model.fit(
            X_train,
            y_train,
        )

        xgb_pred = xgb_model.predict(
            X_test
        )

        xgb_mae = mean_absolute_error(
            y_test,
            xgb_pred,
        )

        xgb_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                xgb_pred,
            )
        )

        xgb_r2 = r2_score(
            y_test,
            xgb_pred,
        )


        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        results.append(
            {
                "FEATURE_SET": feature_set_name,
                "MODEL": "LINEAR",
                "FOLD": fold_number,
                "HELD_OUT_WELL": test_well,
                "N_TRAIN": len(train_df),
                "N_TEST": len(test_df),
                "MAE": linear_mae,
                "RMSE": linear_rmse,
                "R2": linear_r2,
            }
        )

        results.append(
            {
                "FEATURE_SET": feature_set_name,
                "MODEL": "XGBOOST",
                "FOLD": fold_number,
                "HELD_OUT_WELL": test_well,
                "N_TRAIN": len(train_df),
                "N_TEST": len(test_df),
                "MAE": xgb_mae,
                "RMSE": xgb_rmse,
                "R2": xgb_r2,
            }
        )


        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        fold_predictions = test_df[
            [
                WELL_COLUMN,
                "DEPTH",
                TARGET,
            ]
        ].copy()

        fold_predictions[
            "FEATURE_SET"
        ] = feature_set_name

        fold_predictions[
            "FOLD"
        ] = fold_number

        fold_predictions[
            "LINEAR_PRED"
        ] = linear_pred

        fold_predictions[
            "XGB_PRED"
        ] = xgb_pred

        fold_predictions[
            "LINEAR_RESIDUAL"
        ] = (
            y_test.to_numpy()
            - linear_pred
        )

        fold_predictions[
            "XGB_RESIDUAL"
        ] = (
            y_test.to_numpy()
            - xgb_pred
        )

        predictions.append(
            fold_predictions
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    predictions,
    ignore_index=True,
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    results_df
    .groupby(
        [
            "FEATURE_SET",
            "MODEL",
        ]
    )
    .agg(
        MEAN_MAE=("MAE", "mean"),
        MEDIAN_MAE=("MAE", "median"),
        STD_MAE=("MAE", "std"),

        MEAN_RMSE=("RMSE", "mean"),
        MEDIAN_RMSE=("RMSE", "median"),
        STD_RMSE=("RMSE", "std"),

        MEAN_R2=("R2", "mean"),
        MEDIAN_R2=("R2", "median"),
        STD_R2=("R2", "std"),

        MIN_R2=("R2", "min"),
        MAX_R2=("R2", "max"),

        N_WELLS=("HELD_OUT_WELL", "nunique"),
    )
    .reset_index()
)


# ============================================================
# PER-WELL COMPARISON
# ============================================================

per_well_df = (
    results_df
    .pivot_table(
        index=[
            "HELD_OUT_WELL",
        ],
        columns=[
            "FEATURE_SET",
            "MODEL",
        ],
        values=[
            "MAE",
            "RMSE",
            "R2",
        ],
    )
)

per_well_df.columns = [
    "_".join(
        [
            str(x)
            for x in col
        ]
    )
    for col in per_well_df.columns
]

per_well_df = (
    per_well_df
    .reset_index()
)


# ============================================================
# SAVE
# ============================================================

results_path = (
    OUTPUT_DIR
    / "phif_depth_feature_cv_results.csv"
)

summary_path = (
    OUTPUT_DIR
    / "phif_depth_feature_cv_summary.csv"
)

per_well_path = (
    OUTPUT_DIR
    / "phif_depth_feature_cv_per_well.csv"
)

predictions_path = (
    OUTPUT_DIR
    / "phif_depth_feature_cv_predictions.csv"
)


results_df.to_csv(
    results_path,
    index=False,
)

summary_df.to_csv(
    summary_path,
    index=False,
)

per_well_df.to_csv(
    per_well_path,
    index=False,
)

predictions_df.to_csv(
    predictions_path,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GROUP CV SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("PER-WELL RÂ² COMPARISON")
print("=" * 70)

r2_columns = [
    col
    for col in per_well_df.columns
    if col.endswith("_R2")
]

print(
    per_well_df[
        ["HELD_OUT_WELL"] + r2_columns
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_path)
print(summary_path)
print(per_well_path)
print(predictions_path)

print("\nDone.")
