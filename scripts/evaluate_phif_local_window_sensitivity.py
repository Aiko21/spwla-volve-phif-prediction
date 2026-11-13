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

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_local_window_sensitivity.csv"
)

WINDOWS = [5, 11, 21]


FEATURE_FAMILIES = {
    "BASE_GR_LOGRT": [
        "GR",
        "LOG10_RT",
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

def add_local_features(df, window):
    """
    Compute local features independently within each well.

    Only GR, RT, and DEPTH are used.
    PHIF is never used to construct features.
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

    for well, idx in df.groupby("WELL", sort=False).groups.items():

        idx = list(idx)

        gr = df.loc[idx, "GR"]
        logrt = df.loc[idx, "LOG10_RT"]
        depth = df.loc[idx, "DEPTH"]

        # ----------------------------------------------------
        # Local statistics
        # ----------------------------------------------------

        df.loc[idx, "GR_LOCAL_MEAN"] = (
            gr.rolling(
                window=window,
                center=True,
                min_periods=3,
            )
            .mean()
            .to_numpy()
        )

        df.loc[idx, "GR_LOCAL_STD"] = (
            gr.rolling(
                window=window,
                center=True,
                min_periods=3,
            )
            .std()
            .to_numpy()
        )

        df.loc[idx, "LOGRT_LOCAL_MEAN"] = (
            logrt.rolling(
                window=window,
                center=True,
                min_periods=3,
            )
            .mean()
            .to_numpy()
        )

        df.loc[idx, "LOGRT_LOCAL_STD"] = (
            logrt.rolling(
                window=window,
                center=True,
                min_periods=3,
            )
            .std()
            .to_numpy()
        )

        # ----------------------------------------------------
        # Depth gradients
        # ----------------------------------------------------

        depth_values = depth.to_numpy(dtype=float)

        gr_values = gr.to_numpy(dtype=float)
        logrt_values = logrt.to_numpy(dtype=float)

        # Gradient arrays
        gr_gradient = np.full(
            len(idx),
            np.nan,
        )

        logrt_gradient = np.full(
            len(idx),
            np.nan,
        )

        valid_gr = (
            np.isfinite(gr_values)
            & np.isfinite(depth_values)
        )

        if valid_gr.sum() >= 2:
            gr_gradient[valid_gr] = np.gradient(
                gr_values[valid_gr],
                depth_values[valid_gr],
            )

        valid_rt = (
            np.isfinite(logrt_values)
            & np.isfinite(depth_values)
        )

        if valid_rt.sum() >= 2:
            logrt_gradient[valid_rt] = np.gradient(
                logrt_values[valid_rt],
                depth_values[valid_rt],
            )

        df.loc[idx, "GR_GRADIENT"] = gr_gradient
        df.loc[idx, "LOGRT_GRADIENT"] = logrt_gradient

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("PHIF LOCAL WINDOW SENSITIVITY — STRICT LEAVE-ONE-WELL-OUT CV")
print("=" * 100)

print(f"\nDataset: {DATA_PATH}")
print(f"Windows: {WINDOWS}")

df_raw = pd.read_csv(DATA_PATH)

required_columns = {
    "WELL",
    "DEPTH",
    "GR",
    "RT",
    "PHIF",
}

missing = required_columns - set(df_raw.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print(f"Rows: {len(df_raw):,}")
print(f"Wells: {df_raw['WELL'].nunique()}")


# ============================================================
# RUN WINDOW EXPERIMENT
# ============================================================

all_results = []

wells = sorted(df_raw["WELL"].unique())

for window in WINDOWS:

    print("\n" + "=" * 100)
    print(f"WINDOW = {window} SAMPLES")
    print("=" * 100)

    df = add_local_features(
        df_raw,
        window=window,
    )

    df = df[
        df["PHIF"].notna()
        & df["GR"].notna()
        & df["LOG10_RT"].notna()
    ].copy()

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

        for feature_family, features in FEATURE_FAMILIES.items():

            train_model = train.dropna(
                subset=features + ["PHIF"]
            )

            test_model = test.dropna(
                subset=features + ["PHIF"]
            )

            if (
                len(train_model) == 0
                or len(test_model) == 0
            ):
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

            predictions = model.predict(
                X_test
            )

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

            all_results.append(
                {
                    "WINDOW": window,
                    "FOLD": fold,
                    "HELD_OUT_WELL": held_out_well,
                    "FEATURE_SET": feature_family,
                    "N_TRAIN": len(train_model),
                    "N_TEST": len(test_model),
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                }
            )

            print(
                f"window={window:2d} "
                f"| well={held_out_well:<12} "
                f"| {feature_family:<20} "
                f"| MAE={mae:.6f} "
                f"| RMSE={rmse:.6f} "
                f"| R2={r2:.6f}"
            )


results_df = pd.DataFrame(
    all_results
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("WINDOW SENSITIVITY SUMMARY")
print("=" * 100)

summary = (
    results_df
    .groupby(
        ["WINDOW", "FEATURE_SET"]
    )
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
        POSITIVE_R2=(
            "R2",
            lambda x: (x > 0).sum(),
        ),
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# BEST WINDOW BY FEATURE FAMILY
# ============================================================

print("\n" + "=" * 100)
print("BEST WINDOW BY FEATURE FAMILY")
print("=" * 100)

for feature_family in FEATURE_FAMILIES:

    subset = summary[
        summary["FEATURE_SET"]
        == feature_family
    ].copy()

    best = subset.sort_values(
        [
            "MEAN_MAE",
            "MEAN_RMSE",
        ],
        ascending=[
            True,
            True,
        ],
    ).iloc[0]

    print(
        f"\n{feature_family}"
    )

    print(
        f"  Best window       : "
        f"{int(best['WINDOW'])}"
    )

    print(
        f"  Mean MAE          : "
        f"{best['MEAN_MAE']:.6f}"
    )

    print(
        f"  Mean RMSE         : "
        f"{best['MEAN_RMSE']:.6f}"
    )

    print(
        f"  Mean R²           : "
        f"{best['MEAN_R2']:.6f}"
    )

    print(
        f"  Positive R²       : "
        f"{int(best['POSITIVE_R2'])}/9"
    )


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)
print(OUTPUT_PATH)