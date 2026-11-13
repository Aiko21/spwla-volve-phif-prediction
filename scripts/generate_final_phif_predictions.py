
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# ============================================================
# CONFIGURATION — LOCKED PHIF MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/phif_ml_dataset_v1.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports/models"

OOF_PATH = OUTPUT_DIR / "final_phif_oof_predictions.csv"
FULL_PATH = OUTPUT_DIR / "final_phif_full_predictions.csv"
OOF_METRICS_PATH = (
    OUTPUT_DIR / "final_phif_oof_metrics_by_well.csv"
)

WINDOW = 11
RANDOM_STATE = 42


FEATURES = [
    "GR",
    "LOG10_RT",
    "GR_LOCAL_STD_11",
    "LOGRT_LOCAL_STD_11",
]

MODEL_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 30,
    "l2_regularization": 1.0,
    "random_state": RANDOM_STATE,
}


def create_model():
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(data):
    df = data.copy()

    # Stable ordering is essential for vertical rolling features.
    df = df.sort_values(["WELL", "DEPTH"]).reset_index(drop=True)

    # Preserve raw RT; use log10 transform as model feature.
    df["LOG10_RT"] = np.nan
    valid_rt = df["RT"].notna() & (df["RT"] > 0)
    df.loc[valid_rt, "LOG10_RT"] = np.log10(
        df.loc[valid_rt, "RT"]
    )

    grouped = df.groupby("WELL", sort=False)

    df["GR_LOCAL_STD_11"] = grouped["GR"].transform(
        lambda s: s.rolling(
            window=WINDOW, center=True, min_periods=2
        ).std()
    )

    df["LOGRT_LOCAL_STD_11"] = grouped["LOG10_RT"].transform(
        lambda s: s.rolling(
            window=WINDOW, center=True, min_periods=2
        ).std()
    )

    return df


def eligible_rows(df):
    return df.dropna(subset=FEATURES + ["PHIF"]).copy()


def add_prediction_columns(frame, predictions, prediction_type):
    result = frame[
        [
            "WELL",
            "DEPTH",
            "GR",
            "RT",
            "PHIF",
            "LOG10_RT",
            "GR_LOCAL_STD_11",
            "LOGRT_LOCAL_STD_11",
        ]
    ].copy()

    result["PREDICTED_PHIF"] = predictions
    result["RESIDUAL"] = result["PREDICTED_PHIF"] - result["PHIF"]
    result["ABS_ERROR"] = result["RESIDUAL"].abs()
    result["PREDICTION_TYPE"] = prediction_type

    return result


# ============================================================
# LEAVE-ONE-WELL-OUT PREDICTIONS
# ============================================================

def generate_oof(df):
    results = []
    wells = sorted(df["WELL"].dropna().unique())

    print("\nGenerating leave-one-well-out predictions...")

    for held_out in wells:
        train = eligible_rows(df[df["WELL"] != held_out])
        test = eligible_rows(df[df["WELL"] == held_out])

        if train.empty or test.empty:
            raise ValueError(
                f"Empty train/test data for held-out well: {held_out}"
            )

        model = create_model()
        model.fit(train[FEATURES], train["PHIF"])
        pred = model.predict(test[FEATURES])

        result = add_prediction_columns(test, pred, "OOF_LOO")
        result["HELD_OUT_WELL"] = held_out
        results.append(result)

        print(
            f"{held_out}: train={len(train):,}, "
            f"predicted={len(result):,}"
        )

    return pd.concat(results, ignore_index=True)


# ============================================================
# FULL-DATA FITTED MODEL PREDICTIONS
# ============================================================

def generate_full_data_predictions(df):
    eligible = eligible_rows(df)

    model = create_model()
    model.fit(eligible[FEATURES], eligible["PHIF"])
    pred = model.predict(eligible[FEATURES])

    result = add_prediction_columns(
        eligible, pred, "FULL_DATA_FIT"
    )
    result["HELD_OUT_WELL"] = "NONE"

    return result


# ============================================================
# METRICS
# ============================================================

def calculate_oof_metrics(oof):
    rows = []

    for well, group in oof.groupby("WELL", sort=True):
        rows.append({
            "WELL": well,
            "N": len(group),
            "ACTUAL_PHIF_MEAN": group["PHIF"].mean(),
            "PREDICTED_PHIF_MEAN": group["PREDICTED_PHIF"].mean(),
            "MAE": mean_absolute_error(
                group["PHIF"], group["PREDICTED_PHIF"]
            ),
            "RMSE": np.sqrt(mean_squared_error(
                group["PHIF"], group["PREDICTED_PHIF"]
            )),
            "R2": r2_score(
                group["PHIF"], group["PREDICTED_PHIF"]
            ),
            "MEAN_RESIDUAL": group["RESIDUAL"].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("FINAL PHIF PREDICTION DATASET")
    print("=" * 68)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH)

    required = {"WELL", "DEPTH", "GR", "RT", "PHIF"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Input rows: {len(raw):,}")
    print(f"Input wells: {raw['WELL'].nunique()}")

    df = prepare_features(raw)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oof = generate_oof(df)
    full = generate_full_data_predictions(df)
    metrics = calculate_oof_metrics(oof)

    # Save only after all computations have succeeded.
    oof.to_csv(OOF_PATH, index=False)
    full.to_csv(FULL_PATH, index=False)
    metrics.to_csv(OOF_METRICS_PATH, index=False)

    print("\n" + "=" * 68)
    print("OOF METRICS BY WELL")
    print("=" * 68)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n" + "=" * 68)
    print("OUTPUT VALIDATION")
    print("=" * 68)
    print(f"OOF rows: {len(oof):,}")
    print(f"OOF wells: {oof['WELL'].nunique()}")
    print(f"Full-data rows: {len(full):,}")
    print(f"OOF missing predictions: {oof['PREDICTED_PHIF'].isna().sum()}")
    print(f"Full missing predictions: {full['PREDICTED_PHIF'].isna().sum()}")

    print("\nSaved:")
    print(OOF_PATH)
    print(FULL_PATH)
    print(OOF_METRICS_PATH)

    print("\nInterpretation:")
    print("- Use OOF predictions for cross-well evaluation and characterization.")
    print("- Full-data fitted predictions are not independent performance estimates.")
    print("- This script does not modify the source dataset or Git history.")


if __name__ == "__main__":
    main()
