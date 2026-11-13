from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("data/processed/phif_ml_dataset_v1.csv")
OUTPUT_PATH = Path(
    "reports/models/final_phif_development_selection.csv"
)

RANDOM_STATE = 42

DEVELOPMENT_WELLS = [
    "15_9-F-1A",
    "15_9-F-1B",
    "15_9-F11 T2",
    "15_9-F11A",
    "15_9-F11B",
    "15_9-F12",
    "15_9-F15D",
]

HOLDOUT_WELLS = [
    "15_9-F-1",
    "15_9-F1C",
]

WINDOWS = [5, 11, 21]


# ============================================================
# MODEL
# ============================================================

def create_model():
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
    df = df.copy()

    # --------------------------------------------------------
    # Base transformed feature
    # --------------------------------------------------------

    df["LOG10_RT"] = np.log10(
        df["RT"].clip(lower=1e-6)
    )

    # --------------------------------------------------------
    # Rolling local features
    # Calculated independently within each well
    # --------------------------------------------------------

    df[f"GR_LOCAL_MEAN_{window}"] = (
        df.groupby("WELL")["GR"]
        .transform(
            lambda x: x.rolling(
                window=window,
                center=True,
                min_periods=1
            ).mean()
        )
    )

    df[f"LOGRT_LOCAL_MEAN_{window}"] = (
        df.groupby("WELL")["LOG10_RT"]
        .transform(
            lambda x: x.rolling(
                window=window,
                center=True,
                min_periods=1
            ).mean()
        )
    )

    df[f"GR_LOCAL_STD_{window}"] = (
        df.groupby("WELL")["GR"]
        .transform(
            lambda x: x.rolling(
                window=window,
                center=True,
                min_periods=2
            ).std()
        )
    )

    df[f"LOGRT_LOCAL_STD_{window}"] = (
        df.groupby("WELL")["LOG10_RT"]
        .transform(
            lambda x: x.rolling(
                window=window,
                center=True,
                min_periods=2
            ).std()
        )
    )

    # --------------------------------------------------------
    # Depth-based gradients
    # --------------------------------------------------------

    df["GR_GRADIENT"] = np.nan
    df["LOGRT_GRADIENT"] = np.nan

    for well, idx in df.groupby("WELL").groups.items():

        group = df.loc[idx].sort_values("DEPTH")

        depth = group["DEPTH"].to_numpy()
        gr = group["GR"].to_numpy()
        logrt = group["LOG10_RT"].to_numpy()

        gr_gradient = np.gradient(
            gr,
            depth
        )

        logrt_gradient = np.gradient(
            logrt,
            depth
        )

        df.loc[
            group.index,
            "GR_GRADIENT"
        ] = gr_gradient

        df.loc[
            group.index,
            "LOGRT_GRADIENT"
        ] = logrt_gradient

    return df

# ============================================================
# FEATURE SET
# ============================================================

def select_features(df, feature_set, window):

    base = [
        "GR",
        "LOG10_RT",
    ]

    local_variability = [
        f"GR_LOCAL_STD_{window}",
        f"LOGRT_LOCAL_STD_{window}",
    ]

    local_change = [
        "GR_GRADIENT",
        "LOGRT_GRADIENT",
    ]

    all_local = (
        local_variability
        + local_change
        + [
            f"GR_LOCAL_MEAN_{window}",
            f"LOGRT_LOCAL_MEAN_{window}",
        ]
    )

    if feature_set == "BASE":
        return base

    if feature_set == "LOCAL_VARIABILITY":
        return base + local_variability

    if feature_set == "LOCAL_CHANGE":
        return base + local_change

    if feature_set == "ALL_LOCAL":
        return base + all_local

    raise ValueError(
        f"Unknown feature set: {feature_set}"
    )


# ============================================================
# LOO CV ON DEVELOPMENT WELLS
# ============================================================

def development_cv(df, feature_set, window):

    wells = DEVELOPMENT_WELLS
    fold_results = []

    feature_df = add_local_features(
        df,
        window=window
    )

    features = select_features(
        feature_df,
        feature_set,
        window
    )

    for held_out_well in wells:

        train_wells = [
            w for w in wells
            if w != held_out_well
        ]

        train = feature_df[
            feature_df["WELL"].isin(train_wells)
        ].copy()

        test = feature_df[
            feature_df["WELL"] == held_out_well
        ].copy()

        train = train.dropna(
            subset=features + ["PHIF"]
        )

        test = test.dropna(
            subset=features + ["PHIF"]
        )

        model = create_model()

        model.fit(
            train[features],
            train["PHIF"]
        )

        pred = model.predict(
            test[features]
        )

        mae = mean_absolute_error(
            test["PHIF"],
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                test["PHIF"],
                pred
            )
        )

        r2 = r2_score(
            test["PHIF"],
            pred
        )

        fold_results.append({
            "HELD_OUT_WELL": held_out_well,
            "N_TRAIN": len(train),
            "N_TEST": len(test),
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    fold_df = pd.DataFrame(fold_results)

    return {
        "feature_set": feature_set,
        "window": window,
        "mean_mae": fold_df["MAE"].mean(),
        "median_mae": fold_df["MAE"].median(),
        "mean_rmse": fold_df["RMSE"].mean(),
        "median_rmse": fold_df["RMSE"].median(),
        "mean_r2": fold_df["R2"].mean(),
        "median_r2": fold_df["R2"].median(),
        "min_r2": fold_df["R2"].min(),
        "max_r2": fold_df["R2"].max(),
        "positive_r2": int(
            (fold_df["R2"] > 0).sum()
        ),
        "fold_results": fold_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL PHIF DEVELOPMENT-WELL FEATURE SELECTION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    df = df[
        df["WELL"].isin(
            DEVELOPMENT_WELLS + HOLDOUT_WELLS
        )
    ].copy()

    print(f"Dataset rows: {len(df):,}")
    print(f"Development wells: {DEVELOPMENT_WELLS}")
    print(f"Holdout wells: {HOLDOUT_WELLS}")

    results = []
    detailed_results = []

    for window in WINDOWS:

        for feature_set in [
            "BASE",
            "LOCAL_VARIABILITY",
            "LOCAL_CHANGE",
            "ALL_LOCAL",
        ]:

            print(
                f"\nEvaluating "
                f"{feature_set} | window={window}"
            )

            result = development_cv(
                df,
                feature_set,
                window
            )

            results.append({
                "WINDOW": window,
                "FEATURE_SET": feature_set,
                "MEAN_MAE": result["mean_mae"],
                "MEDIAN_MAE": result["median_mae"],
                "MEAN_RMSE": result["mean_rmse"],
                "MEDIAN_RMSE": result["median_rmse"],
                "MEAN_R2": result["mean_r2"],
                "MEDIAN_R2": result["median_r2"],
                "MIN_R2": result["min_r2"],
                "MAX_R2": result["max_r2"],
                "POSITIVE_R2": result["positive_r2"],
            })

            fold_df = result["fold_results"].copy()
            fold_df["WINDOW"] = window
            fold_df["FEATURE_SET"] = feature_set

            detailed_results.append(
                fold_df
            )

    summary_df = pd.DataFrame(results)

    detailed_df = pd.concat(
        detailed_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Selection rule
    # Primary: lowest mean MAE
    # Secondary: lowest mean RMSE
    # Tertiary: highest mean R2
    # --------------------------------------------------------

    ranked = summary_df.sort_values(
        [
            "MEAN_MAE",
            "MEAN_RMSE",
            "MEAN_R2",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    best = ranked.iloc[0]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    detailed_path = OUTPUT_PATH.with_name(
        "final_phif_development_selection_folds.csv"
    )

    detailed_df.to_csv(
        detailed_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("DEVELOPMENT-WELL RESULTS")
    print("=" * 70)

    print(
        ranked.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SELECTED CONFIGURATION")
    print("=" * 70)

    print(
        f"Feature set : {best['FEATURE_SET']}"
    )

    print(
        f"Window      : {int(best['WINDOW'])}"
    )

    print(
        f"Mean MAE    : {best['MEAN_MAE']:.6f}"
    )

    print(
        f"Mean RMSE   : {best['MEAN_RMSE']:.6f}"
    )

    print(
        f"Mean R2     : {best['MEAN_R2']:.6f}"
    )

    print(
        f"Positive R2 : {int(best['POSITIVE_R2'])}/"
        f"{len(DEVELOPMENT_WELLS)}"
    )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(detailed_path)


if __name__ == "__main__":
    main()