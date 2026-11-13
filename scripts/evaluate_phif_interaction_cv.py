from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_RESULTS_PATH = (
    ROOT
    / "reports"
    / "models"
    / "phif_interaction_cv_results.csv"
)

OUTPUT_SUMMARY_PATH = (
    ROOT
    / "reports"
    / "models"
    / "phif_interaction_cv_summary.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

required_cols = [
    "WELL",
    "GR",
    "RT",
    "PHIF",
]

missing = [
    col
    for col in required_cols
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
# FEATURE SETS
# ============================================================

df["GR_X_LOG10_RT"] = (
    df["GR"] * df["LOG10_RT"]
)


FEATURE_SETS = {
    "BASE_GR_LOGRT": [
        "GR",
        "LOG10_RT",
    ],
    "INTERACTION_GR_LOGRT": [
        "GR",
        "LOG10_RT",
        "GR_X_LOG10_RT",
    ],
}


# ============================================================
# MODELS
# ============================================================

models = {
    "LINEAR": Pipeline(
        [
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
}

if XGBRegressor is not None:

    models["XGBOOST"] = XGBRegressor(
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


# ============================================================
# LEAVE-ONE-WELL-OUT CV
# ============================================================

wells = sorted(
    df["WELL"].unique()
)

results = []

print("=" * 110)
print("PHIF INTERACTION FEATURE — LEAVE-ONE-WELL-OUT CV")
print("=" * 110)
print()

print(f"Number of wells: {len(wells)}")
print(f"Total rows     : {len(df):,}")
print()


for fold, held_out_well in enumerate(
    wells,
    start=1,
):

    train_df = df[
        df["WELL"] != held_out_well
    ].copy()

    test_df = df[
        df["WELL"] == held_out_well
    ].copy()

    y_train = train_df["PHIF"]
    y_test = test_df["PHIF"]

    print(
        f"Fold {fold:02d} | "
        f"Held-out: {held_out_well} | "
        f"Train: {len(train_df):,} | "
        f"Test: {len(test_df):,}"
    )

    for feature_set_name, feature_cols in FEATURE_SETS.items():

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_test
            )

            mae = mean_absolute_error(
                y_test,
                pred,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred,
                )
            )

            r2 = r2_score(
                y_test,
                pred,
            )

            results.append(
                {
                    "FOLD": fold,
                    "HELD_OUT_WELL": held_out_well,
                    "N_TRAIN": len(train_df),
                    "N_TEST": len(test_df),
                    "FEATURE_SET": feature_set_name,
                    "MODEL": model_name,
                    "ACTUAL_PHIF_MEAN": y_test.mean(),
                    "ACTUAL_PHIF_MEDIAN": y_test.median(),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# PRINT RAW RESULTS
# ============================================================

print()
print("=" * 110)
print("PER-WELL RESULTS")
print("=" * 110)
print()

print(
    results_df[
        [
            "HELD_OUT_WELL",
            "FEATURE_SET",
            "MODEL",
            "MAE",
            "RMSE",
            "R2",
        ]
    ].to_string(index=False)
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


print()
print("=" * 110)
print("CV SUMMARY")
print("=" * 110)
print()

print(
    summary_df.to_string(index=False)
)


# ============================================================
# BASELINE VS INTERACTION
# ============================================================

print()
print("=" * 110)
print("INTERACTION INCREMENTAL VALUE BY WELL")
print("=" * 110)
print()


for model_name in models.keys():

    base = results_df[
        (
            results_df["FEATURE_SET"]
            == "BASE_GR_LOGRT"
        )
        &
        (
            results_df["MODEL"]
            == model_name
        )
    ].copy()

    interaction = results_df[
        (
            results_df["FEATURE_SET"]
            == "INTERACTION_GR_LOGRT"
        )
        &
        (
            results_df["MODEL"]
            == model_name
        )
    ].copy()

    base = base.rename(
        columns={
            "MAE": "BASE_MAE",
            "RMSE": "BASE_RMSE",
            "R2": "BASE_R2",
        }
    )

    interaction = interaction.rename(
        columns={
            "MAE": "INT_MAE",
            "RMSE": "INT_RMSE",
            "R2": "INT_R2",
        }
    )

    comparison = base[
        [
            "FOLD",
            "HELD_OUT_WELL",
            "BASE_MAE",
            "BASE_RMSE",
            "BASE_R2",
        ]
    ].merge(
        interaction[
            [
                "FOLD",
                "HELD_OUT_WELL",
                "INT_MAE",
                "INT_RMSE",
                "INT_R2",
            ]
        ],
        on=[
            "FOLD",
            "HELD_OUT_WELL",
        ],
        how="inner",
    )

    comparison["DELTA_MAE"] = (
        comparison["BASE_MAE"]
        - comparison["INT_MAE"]
    )

    comparison["DELTA_RMSE"] = (
        comparison["BASE_RMSE"]
        - comparison["INT_RMSE"]
    )

    comparison["DELTA_R2"] = (
        comparison["INT_R2"]
        - comparison["BASE_R2"]
    )

    comparison["MAE_IMPROVED"] = (
        comparison["DELTA_MAE"] > 0
    )

    comparison["RMSE_IMPROVED"] = (
        comparison["DELTA_RMSE"] > 0
    )

    comparison["R2_IMPROVED"] = (
        comparison["DELTA_R2"] > 0
    )

    print()
    print(f"MODEL: {model_name}")
    print()

    print(
        comparison[
            [
                "HELD_OUT_WELL",
                "BASE_MAE",
                "INT_MAE",
                "DELTA_MAE",
                "BASE_RMSE",
                "INT_RMSE",
                "DELTA_RMSE",
                "BASE_R2",
                "INT_R2",
                "DELTA_R2",
            ]
        ].to_string(index=False)
    )

    print()

    print(
        "MAE improved : "
        f"{comparison['MAE_IMPROVED'].sum()}"
        f"/{len(comparison)}"
    )

    print(
        "RMSE improved: "
        f"{comparison['RMSE_IMPROVED'].sum()}"
        f"/{len(comparison)}"
    )

    print(
        "R2 improved  : "
        f"{comparison['R2_IMPROVED'].sum()}"
        f"/{len(comparison)}"
    )

    print()

    print(
        "Mean ΔMAE : "
        f"{comparison['DELTA_MAE'].mean():.6f}"
    )

    print(
        "Mean ΔRMSE: "
        f"{comparison['DELTA_RMSE'].mean():.6f}"
    )

    print(
        "Mean ΔR2  : "
        f"{comparison['DELTA_R2'].mean():.6f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_RESULTS_PATH,
    index=False,
)

summary_df.to_csv(
    OUTPUT_SUMMARY_PATH,
    index=False,
)


print()
print("=" * 110)
print("OUTPUT")
print("=" * 110)
print(OUTPUT_RESULTS_PATH)
print(OUTPUT_SUMMARY_PATH)
print()
print("Done.")