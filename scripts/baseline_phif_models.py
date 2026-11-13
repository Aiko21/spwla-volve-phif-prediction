from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "models"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    DATA_DIR / "phif_train.csv"
)

validation = pd.read_csv(
    DATA_DIR / "phif_validation.csv"
)

test = pd.read_csv(
    DATA_DIR / "phif_test.csv"
)


print("=" * 70)
print("PHIF BASELINE MODELING V1")
print("=" * 70)

print(
    f"\nTrain      : {len(train):,} rows"
)

print(
    f"Validation : {len(validation):,} rows"
)

print(
    f"Test       : {len(test):,} rows"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

for df in [
    train,
    validation,
    test,
]:

    df["LOG10_RT"] = np.log10(
        df["RT"]
    )


TARGET = "PHIF"


FEATURE_CONFIGS = {
    "GR_only": [
        "GR"
    ],

    "GR_RT": [
        "GR",
        "RT"
    ],

    "GR_LOG10RT": [
        "GR",
        "LOG10_RT"
    ],

    "GR_RT_LOG10RT": [
        "GR",
        "RT",
        "LOG10_RT"
    ],
}


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "DummyMedian": DummyRegressor(
        strategy="median"
    ),

    "LinearRegression": Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            ),
        ]
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# PREPARE TARGETS
# ============================================================

y_train = train[TARGET]

y_validation = validation[TARGET]

y_test = test[TARGET]


# ============================================================
# VALIDATION EXPERIMENTS
# ============================================================

validation_results = []

print("\n" + "=" * 70)
print("VALIDATION EXPERIMENTS")
print("=" * 70)


for feature_name, features in FEATURE_CONFIGS.items():

    X_train = train[features]

    X_validation = validation[features]

    print(
        f"\nFeature set: {feature_name}"
    )

    print(
        f"Features: {features}"
    )

    for model_name, model in MODELS.items():

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_validation
        )

        metrics = calculate_metrics(
            y_validation,
            prediction
        )

        row = {
            "MODEL": model_name,
            "FEATURE_SET": feature_name,
            "FEATURES": ", ".join(features),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }

        validation_results.append(
            row
        )

        print(
            f"{model_name:18s} "
            f"MAE={metrics['MAE']:.5f} "
            f"RMSE={metrics['RMSE']:.5f} "
            f"R2={metrics['R2']:.5f}"
        )


validation_results_df = pd.DataFrame(
    validation_results
)

validation_results_df = (
    validation_results_df
    .sort_values(
        "RMSE",
        ascending=True
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE VALIDATION RESULTS
# ============================================================

validation_path = (
    REPORT_DIR
    / "phif_baseline_validation_results.csv"
)

validation_results_df.to_csv(
    validation_path,
    index=False
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_row = (
    validation_results_df
    .iloc[0]
)

best_model_name = (
    best_row["MODEL"]
)

best_feature_set = (
    best_row["FEATURE_SET"]
)

best_features = (
    FEATURE_CONFIGS[
        best_feature_set
    ]
)


print("\n" + "=" * 70)
print("VALIDATION RANKING")
print("=" * 70)

print(
    validation_results_df[
        [
            "MODEL",
            "FEATURE_SET",
            "MAE",
            "RMSE",
            "R2",
        ]
    ]
    .round(6)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("SELECTED BASELINE")
print("=" * 70)

print(
    f"Model       : {best_model_name}"
)

print(
    f"Feature set : {best_feature_set}"
)

print(
    f"Features    : {best_features}"
)

print(
    f"Validation MAE  : "
    f"{best_row['MAE']:.6f}"
)

print(
    f"Validation RMSE : "
    f"{best_row['RMSE']:.6f}"
)

print(
    f"Validation R2   : "
    f"{best_row['R2']:.6f}"
)


# ============================================================
# FINAL REFIT
# ============================================================

print("\n" + "=" * 70)
print("FINAL REFIT: TRAIN + VALIDATION")
print("=" * 70)


train_validation = pd.concat(
    [
        train,
        validation,
    ],
    ignore_index=True
)


X_train_validation = (
    train_validation[
        best_features
    ]
)

y_train_validation = (
    train_validation[TARGET]
)

X_test = (
    test[
        best_features
    ]
)


# Create fresh model instance
if best_model_name == "DummyMedian":

    final_model = DummyRegressor(
        strategy="median"
    )

elif best_model_name == "LinearRegression":

    final_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            ),
        ]
    )

elif best_model_name == "RandomForest":

    final_model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

else:

    raise ValueError(
        f"Unknown model: {best_model_name}"
    )


final_model.fit(
    X_train_validation,
    y_train_validation
)


# ============================================================
# FINAL TEST
# ============================================================

test_prediction = final_model.predict(
    X_test
)

test_metrics = calculate_metrics(
    y_test,
    test_prediction
)


print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(
    f"MAE  : {test_metrics['MAE']:.6f}"
)

print(
    f"RMSE : {test_metrics['RMSE']:.6f}"
)

print(
    f"R2   : {test_metrics['R2']:.6f}"
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions = test[
    [
        "WELL",
        "DEPTH",
        "GR",
        "RT",
        "PHIF",
    ]
].copy()

test_predictions[
    "PHIF_PRED"
] = test_prediction

test_predictions[
    "ABS_ERROR"
] = (
    test_predictions["PHIF"]
    - test_predictions["PHIF_PRED"]
).abs()


prediction_path = (
    REPORT_DIR
    / "phif_final_test_predictions.csv"
)

test_predictions.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# PER-WELL TEST PERFORMANCE
# ============================================================

well_results = []

for well, group in test_predictions.groupby(
    "WELL"
):

    metrics = calculate_metrics(
        group["PHIF"],
        group["PHIF_PRED"]
    )

    well_results.append(
        {
            "WELL": well,
            "ROWS": len(group),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
        }
    )


well_results_df = pd.DataFrame(
    well_results
)

well_results_path = (
    REPORT_DIR
    / "phif_final_test_per_well.csv"
)

well_results_df.to_csv(
    well_results_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "MODEL": best_model_name,
            "FEATURE_SET": best_feature_set,
            "FEATURES": ", ".join(
                best_features
            ),
            "VALIDATION_MAE": best_row["MAE"],
            "VALIDATION_RMSE": best_row["RMSE"],
            "VALIDATION_R2": best_row["R2"],
            "TEST_MAE": test_metrics["MAE"],
            "TEST_RMSE": test_metrics["RMSE"],
            "TEST_R2": test_metrics["R2"],
        }
    ]
)

summary_path = (
    REPORT_DIR
    / "phif_final_model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    validation_path
)

print(
    prediction_path
)

print(
    well_results_path
)

print(
    summary_path
)

print("\nDONE.")
