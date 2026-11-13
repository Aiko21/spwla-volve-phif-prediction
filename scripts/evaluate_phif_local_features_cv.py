from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "phif_ml_dataset_v1.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_local_features_cv_results.csv"
)

# Local window in number of samples.
# 11 samples is deliberately modest and avoids excessive smoothing.
LOCAL_WINDOW = 11

FEATURE_SETS = {
    "BASE_GR_LOGRT": [
        "GR",
        "LOG10_RT",
    ],

    "LOCAL_LEVEL": [
        "GR",
        "LOG10_RT",
        "GR_LOCAL_MEAN",
        "LOGRT_LOCAL_MEAN",
    ],

    "LOCAL_VARIABILITY": [
        "GR",
        "LOG10_RT",
        "GR_LOCAL_STD",
        "LOGRT_LOCAL_STD",
    ],

    "LOCAL_CHANGE": [
        "GR",
        "LOG10_RT",
        "GR_GRADIENT",
        "LOGRT_GRADIENT",
    ],

    "ALL_LOCAL": [
        "GR",
        "LOG10_RT",
        "GR_LOCAL_MEAN",
        "LOGRT_LOCAL_MEAN",
        "GR_LOCAL_STD",
        "LOGRT_LOCAL_STD",
        "GR_GRADIENT",
        "LOGRT_GRADIENT",
    ],
}


# ============================================================
# MODEL
# ============================================================

def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_local_features(df):
    """
    Create local features independently within each well.

    IMPORTANT:
    - Only input logs are used.
    - PHIF is never used to construct features.
    - Each well is processed independently.
    """

    df = df.copy()

    df["LOG10_RT"] = np.where(
        df["RT"] > 0,
        np.log10(df["RT"]),
        np.nan,
    )

    df = df.sort_values(
        ["WELL", "DEPTH"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Rolling local statistics
    # --------------------------------------------------------

    for well, idx in df.groupby("WELL", sort=False).groups.items():

        idx = list(idx)

        gr = df.loc[idx, "GR"]
        logrt = df.loc[idx, "LOG10_RT"]

        # Centered rolling window.
        # This is appropriate for offline well-log interpretation,
        # where the complete well log is available.
        df.loc[idx, "GR_LOCAL_MEAN"] = (
            gr.rolling(
                window=LOCAL_WINDOW,
                center=True,
                min_periods=3,
            )
            .mean()
            .to_numpy()
        )

        df.loc[idx, "GR_LOCAL_STD"] = (
            gr.rolling(
                window=LOCAL_WINDOW,
                center=True,
                min_periods=3,
            )
            .std()
            .to_numpy()
        )

        df.loc[idx, "LOGRT_LOCAL_MEAN"] = (
            logrt.rolling(
                window=LOCAL_WINDOW,
                center=True,
                min_periods=3,
            )
            .mean()
            .to_numpy()
        )

        df.loc[idx, "LOGRT_LOCAL_STD"] = (
            logrt.rolling(
                window=LOCAL_WINDOW,
                center=True,
                min_periods=3,
            )
            .std()
            .to_numpy()
        )

        # ----------------------------------------------------
        # Depth-aware gradients
        # ----------------------------------------------------

        depth = df.loc[idx, "DEPTH"].to_numpy(dtype=float)

        gr_values = gr.to_numpy(dtype=float)
        logrt_values = logrt.to_numpy(dtype=float)

        valid_gr = np.isfinite(gr_values) & np.isfinite(depth)

        gr_gradient = np.full(len(idx), np.nan)

        if valid_gr.sum() >= 2:
            gr_gradient[valid_gr] = np.gradient(
                gr_values[valid_gr],
                depth[valid_gr],
            )

        valid_rt = np.isfinite(logrt_values) & np.isfinite(depth)

        logrt_gradient = np.full(len(idx), np.nan)

        if valid_rt.sum() >= 2:
            logrt_gradient[valid_rt] = np.gradient(
                logrt_values[valid_rt],
                depth[valid_rt],
            )

        df.loc[idx, "GR_GRADIENT"] = gr_gradient
        df.loc[idx, "LOGRT_GRADIENT"] = logrt_gradient

    return df


# ============================================================
# MAIN
# ============================================================

print("=" * 100)
print("PHIF LOCAL FEATURE ENGINEERING — STRICT LEAVE-ONE-WELL-OUT CV")
print("=" * 100)

print(f"\nDataset: {DATA_PATH}")
print(f"Local window: {LOCAL_WINDOW} samples")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Wells: {df['WELL'].nunique()}")

required_columns = {
    "WELL",
    "DEPTH",
    "GR",
    "RT",
    "PHIF",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

df = add_local_features(df)

print("\nLocal feature availability:")

local_columns = [
    "GR_LOCAL_MEAN",
    "GR_LOCAL_STD",
    "LOGRT_LOCAL_MEAN",
    "LOGRT_LOCAL_STD",
    "GR_GRADIENT",
    "LOGRT_GRADIENT",
]

for col in local_columns:
    valid_pct = df[col].notna().mean() * 100
    print(f"{col:<22} valid = {valid_pct:7.2f}%")

# ------------------------------------------------------------
# Only rows with valid target and base inputs
# ------------------------------------------------------------

df = df[
    df["PHIF"].notna()
    & df["GR"].notna()
    & df["LOG10_RT"].notna()
].copy()

wells = sorted(df["WELL"].unique())

print(f"\nML-ready rows: {len(df):,}")
print(f"Wells: {len(wells)}")

# ------------------------------------------------------------
# LOO CV
# ------------------------------------------------------------

results = []

for fold, held_out_well in enumerate(wells, start=1):

    train = df[df["WELL"] != held_out_well].copy()
    test = df[df["WELL"] == held_out_well].copy()

    print(
        f"\nFold {fold}/{len(wells)} "
        f"| Held-out well: {held_out_well}"
    )

    for feature_set_name, features in FEATURE_SETS.items():

        train_model = train.dropna(
            subset=features + ["PHIF"]
        )

        test_model = test.dropna(
            subset=features + ["PHIF"]
        )

        if len(train_model) == 0 or len(test_model) == 0:
            print(
                f"  {feature_set_name:<22} "
                f"SKIPPED — insufficient valid rows"
            )
            continue

        X_train = train_model[features]
        y_train = train_model["PHIF"]

        X_test = test_model[features]
        y_test = test_model["PHIF"]

        model = make_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append(
            {
                "FOLD": fold,
                "HELD_OUT_WELL": held_out_well,
                "FEATURE_SET": feature_set_name,
                "N_TRAIN": len(train_model),
                "N_TEST": len(test_model),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        print(
            f"  {feature_set_name:<22} "
            f"MAE={mae:.6f} "
            f"RMSE={rmse:.6f} "
            f"R2={r2:.6f}"
        )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 100)
print("OVERALL FEATURE-SET PERFORMANCE")
print("=" * 100)

summary = (
    results_df
    .groupby("FEATURE_SET")
    .agg(
        N_WELLS=("HELD_OUT_WELL", "nunique"),
        MEAN_MAE=("MAE", "mean"),
        MEDIAN_MAE=("MAE", "median"),
        MEAN_RMSE=("RMSE", "mean"),
        MEDIAN_RMSE=("RMSE", "median"),
        MEAN_R2=("R2", "mean"),
        MEDIAN_R2=("R2", "median"),
        MIN_R2=("R2", "min"),
        MAX_R2=("R2", "max"),
        POSITIVE_R2=("R2", lambda x: (x > 0).sum()),
    )
    .sort_values(
        "MEAN_MAE"
    )
)

print(
    summary.to_string(
        float_format=lambda x: f"{x:.6f}"
    )
)

# ============================================================
# DELTAS AGAINST BASELINE
# ============================================================

print("\n" + "=" * 100)
print("DELTA AGAINST BASELINE: BASE_GR_LOGRT")
print("=" * 100)

baseline = results_df[
    results_df["FEATURE_SET"] == "BASE_GR_LOGRT"
].set_index("HELD_OUT_WELL")

for feature_set in FEATURE_SETS:

    if feature_set == "BASE_GR_LOGRT":
        continue

    current = results_df[
        results_df["FEATURE_SET"] == feature_set
    ].set_index("HELD_OUT_WELL")

    common = baseline.index.intersection(
        current.index
    )

    delta_mae = (
        current.loc[common, "MAE"]
        - baseline.loc[common, "MAE"]
    )

    delta_rmse = (
        current.loc[common, "RMSE"]
        - baseline.loc[common, "RMSE"]
    )

    delta_r2 = (
        current.loc[common, "R2"]
        - baseline.loc[common, "R2"]
    )

    print(f"\n{feature_set}")

    print(
        f"  Δ Mean MAE  : {delta_mae.mean():+.6f}"
    )

    print(
        f"  Δ Mean RMSE : {delta_rmse.mean():+.6f}"
    )

    print(
        f"  Δ Mean R²   : {delta_r2.mean():+.6f}"
    )

    print(
        f"  MAE improved: {(delta_mae < 0).sum()}/{len(common)} wells"
    )

    print(
        f"  RMSE improved: {(delta_rmse < 0).sum()}/{len(common)} wells"
    )

    print(
        f"  R² improved: {(delta_r2 > 0).sum()}/{len(common)} wells"
    )


print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)
print(OUTPUT_PATH)