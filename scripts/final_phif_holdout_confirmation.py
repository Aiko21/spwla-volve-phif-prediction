from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/processed/phif_ml_dataset_v1.csv"
)

OUTPUT_PATH = Path(
    "reports/models/final_phif_holdout_confirmation.csv"
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

WINDOW = 11


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

def add_local_features(df):

    df = df.copy()

    df["LOG10_RT"] = np.log10(
        df["RT"].clip(lower=1e-6)
    )

    # --------------------------------------------------------
    # Local variability
    # --------------------------------------------------------

    df["GR_LOCAL_STD_11"] = (
        df.groupby("WELL")["GR"]
        .transform(
            lambda x: x.rolling(
                window=11,
                center=True,
                min_periods=2,
            ).std()
        )
    )

    df["LOGRT_LOCAL_STD_11"] = (
        df.groupby("WELL")["LOG10_RT"]
        .transform(
            lambda x: x.rolling(
                window=11,
                center=True,
                min_periods=2,
            ).std()
        )
    )

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model_name,
    features,
    train,
    test,
):

    model = create_model()

    model.fit(
        train[features],
        train["PHIF"],
    )

    prediction = model.predict(
        test[features]
    )

    result = test[
        [
            "WELL",
            "DEPTH",
            "PHIF",
        ]
    ].copy()

    result["PRED_PHIF"] = prediction
    result["RESIDUAL"] = (
        result["PRED_PHIF"]
        - result["PHIF"]
    )
    result["ABS_ERROR"] = (
        result["RESIDUAL"].abs()
    )

    metrics = {
        "MODEL": model_name,
        "MAE": mean_absolute_error(
            result["PHIF"],
            result["PRED_PHIF"],
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                result["PHIF"],
                result["PRED_PHIF"],
            )
        ),
        "R2": r2_score(
            result["PHIF"],
            result["PRED_PHIF"],
        ),
    }

    return metrics, result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL PHIF HOLDOUT CONFIRMATION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    df = df[
        df["WELL"].isin(
            DEVELOPMENT_WELLS
            + HOLDOUT_WELLS
        )
    ].copy()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Development wells: "
        f"{len(DEVELOPMENT_WELLS)}"
    )

    print(
        f"Holdout wells: "
        f"{len(HOLDOUT_WELLS)}"
    )

    df = add_local_features(df)

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    base_features = [
        "GR",
        "LOG10_RT",
    ]

    local_features = [
        "GR",
        "LOG10_RT",
        "GR_LOCAL_STD_11",
        "LOGRT_LOCAL_STD_11",
    ]

    train = df[
        df["WELL"].isin(
            DEVELOPMENT_WELLS
        )
    ].copy()

    test = df[
        df["WELL"].isin(
            HOLDOUT_WELLS
        )
    ].copy()

    results = []
    predictions = []

    # --------------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------------

    train_base = train.dropna(
        subset=base_features + ["PHIF"]
    )

    test_base = test.dropna(
        subset=base_features + ["PHIF"]
    )

    metrics, pred = evaluate(
        "BASE_HGB",
        base_features,
        train_base,
        test_base,
    )

    results.append(metrics)
    predictions.append(
        pred.assign(
            MODEL="BASE_HGB"
        )
    )

    # --------------------------------------------------------
    # LOCAL VARIABILITY MODEL
    # --------------------------------------------------------

    train_local = train.dropna(
        subset=local_features + ["PHIF"]
    )

    test_local = test.dropna(
        subset=local_features + ["PHIF"]
    )

    metrics, pred = evaluate(
        "LOCAL_VARIABILITY_HGB_W11",
        local_features,
        train_local,
        test_local,
    )

    results.append(metrics)
    predictions.append(
        pred.assign(
            MODEL="LOCAL_VARIABILITY_HGB_W11"
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        results
    )

    prediction_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    prediction_path = OUTPUT_PATH.with_name(
        "final_phif_holdout_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HOLDOUT RESULTS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("RESULT BY HOLDOUT WELL")
    print("=" * 70)

    for model_name in prediction_df["MODEL"].unique():

        subset = prediction_df[
            prediction_df["MODEL"]
            == model_name
        ]

        for well in HOLDOUT_WELLS:

            well_data = subset[
                subset["WELL"] == well
            ]

            mae = mean_absolute_error(
                well_data["PHIF"],
                well_data["PRED_PHIF"],
            )

            rmse = np.sqrt(
                mean_squared_error(
                    well_data["PHIF"],
                    well_data["PRED_PHIF"],
                )
            )

            r2 = r2_score(
                well_data["PHIF"],
                well_data["PRED_PHIF"],
            )

            print(
                f"{model_name:30s} "
                f"{well:15s} "
                f"MAE={mae:.6f} "
                f"RMSE={rmse:.6f} "
                f"R2={r2:.6f}"
            )

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(prediction_path)


if __name__ == "__main__":
    main()