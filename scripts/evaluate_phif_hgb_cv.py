from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_RESULTS = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_cv_results.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_cv_summary.csv"
)


RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["LOG10_RT"] = np.log10(
    df["RT"].clip(lower=1e-6)
)

df = df.dropna(
    subset=[
        "WELL",
        "GR",
        "LOG10_RT",
        "PHIF",
    ]
).copy()


FEATURES = [
    "GR",
    "LOG10_RT",
]

TARGET = "PHIF"


# ============================================================
# MODELS
# ============================================================

models = {
    "LINEAR": Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    ),

    "XGBOOST": None,

    "HGB": HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
}


# Import XGBoost only where needed.
from xgboost import XGBRegressor

models["XGBOOST"] = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# LEAVE-ONE-WELL-OUT CV
# ============================================================

wells = sorted(
    df["WELL"].unique()
)

results = []

print("=" * 80)
print("PHIF HGB — STRICT LEAVE-ONE-WELL-OUT CV")
print("=" * 80)

print(f"\nNumber of wells: {len(wells)}")
print(f"Features: {FEATURES}")

for fold, held_out_well in enumerate(
    wells,
    start=1,
):

    train = df[
        df["WELL"] != held_out_well
    ].copy()

    test = df[
        df["WELL"] == held_out_well
    ].copy()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(
        f"\nFold {fold}: "
        f"held-out = {held_out_well}"
    )

    for model_name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        prediction = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            prediction,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prediction,
            )
        )

        r2 = r2_score(
            y_test,
            prediction,
        )

        results.append(
            {
                "FOLD": fold,
                "HELD_OUT_WELL": held_out_well,
                "N_TRAIN": len(train),
                "N_TEST": len(test),
                "MODEL": model_name,
                "ACTUAL_PHIF_MEAN": y_test.mean(),
                "ACTUAL_PHIF_MEDIAN": y_test.median(),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"  {model_name:8s} "
            f"MAE={mae:.6f} "
            f"RMSE={rmse:.6f} "
            f"R2={r2:.6f}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df
    .groupby("MODEL")
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
# SAVE
# ============================================================

OUTPUT_RESULTS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(OUTPUT_RESULTS)
print(OUTPUT_SUMMARY)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)