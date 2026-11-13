from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "models"
    / "phif_relative_feature_cv_results.csv"
)

SUMMARY_PATH = (
    ROOT
    / "reports"
    / "models"
    / "phif_relative_feature_cv_summary.csv"
)


# ==============================================================
# Configuration
# ==============================================================

BASE_FEATURES = [
    "GR",
    "LOG10_RT",
]

ENHANCED_FEATURES = [
    "GR",
    "LOG10_RT",
    "GR_ZWELL",
    "LOGRT_ZWELL",
]


def add_training_based_relative_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Strict blind-well normalization.

    Statistics are calculated ONLY from training wells.

    The held-out well does not contribute to mean/std estimation.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    gr_mean = train_df["GR"].mean()
    gr_std = train_df["GR"].std()

    logrt_mean = train_df["LOG10_RT"].mean()
    logrt_std = train_df["LOG10_RT"].std()

    train_df["GR_ZWELL"] = (
        train_df["GR"] - gr_mean
    ) / gr_std

    test_df["GR_ZWELL"] = (
        test_df["GR"] - gr_mean
    ) / gr_std

    train_df["LOGRT_ZWELL"] = (
        train_df["LOG10_RT"] - logrt_mean
    ) / logrt_std

    test_df["LOGRT_ZWELL"] = (
        test_df["LOG10_RT"] - logrt_mean
    ) / logrt_std

    return train_df, test_df


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions),
    }


def build_linear():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_xgb():
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )


# ==============================================================
# Load data
# ==============================================================

print("=" * 78)
print("STRICT BLIND-WELL RELATIVE FEATURE GROUP CV")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

df["LOG10_RT"] = np.log10(df["RT"])

wells = sorted(df["WELL"].unique())

print()
print(f"Dataset shape: {df.shape}")
print(f"Wells: {len(wells)}")

results = []


# ==============================================================
# Leave-One-Well-Out CV
# ==============================================================

for fold, held_out_well in enumerate(wells, start=1):

    train_df = df[df["WELL"] != held_out_well].copy()
    test_df = df[df["WELL"] == held_out_well].copy()

    # ----------------------------------------------------------
    # IMPORTANT:
    # relative feature parameters are calculated ONLY from train
    # ----------------------------------------------------------

    train_df, test_df = add_training_based_relative_features(
        train_df,
        test_df,
    )

    y_train = train_df["PHIF"]
    y_test = test_df["PHIF"]

    feature_sets = {
        "BASE_GR_LOGRT": BASE_FEATURES,
        "ENHANCED_GLOBAL_TRAIN_STATS": ENHANCED_FEATURES,
    }

    models = {
        "LINEAR": build_linear(),
        "XGBOOST": build_xgb(),
    }

    for feature_set_name, features in feature_sets.items():

        X_train = train_df[features]
        X_test = test_df[features]

        for model_name, model in models.items():

            metrics = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            results.append(
                {
                    "FEATURE_SET": feature_set_name,
                    "MODEL": model_name,
                    "FOLD": fold,
                    "HELD_OUT_WELL": held_out_well,
                    "N_TRAIN": len(train_df),
                    "N_TEST": len(test_df),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )

            print(
                f"Fold {fold:02d} | "
                f"{held_out_well:15s} | "
                f"{feature_set_name:28s} | "
                f"{model_name:8s} | "
                f"MAE={metrics['MAE']:.5f} | "
                f"RMSE={metrics['RMSE']:.5f} | "
                f"R2={metrics['R2']:.5f}"
            )


results_df = pd.DataFrame(results)


# ==============================================================
# Summary
# ==============================================================

summary = (
    results_df
    .groupby(["FEATURE_SET", "MODEL"])
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


# ==============================================================
# Print summary
# ==============================================================

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)

print(summary.to_string(index=False))


# ==============================================================
# Save
# ==============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(OUTPUT_PATH)
print(SUMMARY_PATH)

print()
print("Done.")