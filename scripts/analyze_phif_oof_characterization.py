
from pathlib import Path

import numpy as np
import pandas as pd



# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports/models/final_phif_oof_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports/models"

WELL_OUTPUT = OUTPUT_DIR / (
    "phif_oof_characterization_by_well.csv"
)

DEPTH_OUTPUT = OUTPUT_DIR / (
    "phif_oof_characterization_by_depth_window.csv"
)

WINDOW_SIZE = 10.0  # depth units; verify units before interpretation



# ============================================================
# VALIDATION
# ============================================================

def validate_data(df):
    required = [
        "WELL",
        "DEPTH",
        "PHIF",
        "PREDICTED_PHIF",
        "RESIDUAL",
        "ABS_ERROR",
        "PREDICTION_TYPE",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Required columns missing: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    numeric_columns = [
        "DEPTH",
        "PHIF",
        "PREDICTED_PHIF",
        "RESIDUAL",
        "ABS_ERROR",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[["WELL"] + numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values found in required columns."
        )

    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError(
            "Infinite values found in numeric columns."
        )

    if df["WELL"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty WELL identifiers found.")

    if df.duplicated(["WELL", "DEPTH"]).any():
        raise ValueError(
            "Duplicate WELL + DEPTH rows found. Investigate before aggregation."
        )

    if not df["PREDICTION_TYPE"].eq("OOF_LOO").all():
        raise ValueError(
            "Input contains missing or non-OOF predictions."
        )

    expected_residual = (
        df["PREDICTED_PHIF"] - df["PHIF"]
    )

    if not np.allclose(
        df["RESIDUAL"],
        expected_residual,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(
            "RESIDUAL is inconsistent with PREDICTED_PHIF - PHIF."
        )

    if not np.allclose(
        df["ABS_ERROR"],
        expected_residual.abs(),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(
            "ABS_ERROR is inconsistent with absolute prediction error."
        )

# ============================================================
# WELL-LEVEL CHARACTERIZATION
# ============================================================

def summarize_by_well(df):
    rows = []

    for well, group in df.groupby("WELL", sort=True):
        actual = group["PHIF"]
        predicted = group["PREDICTED_PHIF"]
        residual = predicted - actual

        rows.append({
            "WELL": well,
            "N_SAMPLES": len(group),
            "DEPTH_MIN": group["DEPTH"].min(),
            "DEPTH_MAX": group["DEPTH"].max(),
            "ACTUAL_PHIF_MEAN": actual.mean(),
            "ACTUAL_PHIF_MEDIAN": actual.median(),
            "ACTUAL_PHIF_P10": actual.quantile(0.10),
            "ACTUAL_PHIF_P90": actual.quantile(0.90),
            "PREDICTED_PHIF_MEAN": predicted.mean(),
            "PREDICTED_PHIF_MEDIAN": predicted.median(),
            "PREDICTED_PHIF_P10": predicted.quantile(0.10),
            "PREDICTED_PHIF_P90": predicted.quantile(0.90),
            "MEAN_BIAS_PRED_MINUS_ACTUAL": residual.mean(),
            "MAE": residual.abs().mean(),
            "RMSE": np.sqrt(np.mean(residual ** 2)),
            "ACTUAL_STD": actual.std(),
            "PREDICTED_STD": predicted.std(),
            "STD_RATIO_PRED_TO_ACTUAL": (
                predicted.std() / actual.std()
                if actual.std() > 0 else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# DEPTH-WINDOW CHARACTERIZATION
# ============================================================

def summarize_by_depth_window(df):
    data = df.copy()

    # Fixed-width bins anchored at the floor of the depth minimum.
    depth_origin = np.floor(data["DEPTH"].min() / WINDOW_SIZE) * WINDOW_SIZE

    data["DEPTH_WINDOW_START"] = (
        depth_origin
        + np.floor(
            (data["DEPTH"] - depth_origin) / WINDOW_SIZE
        ) * WINDOW_SIZE
    )

    data["DEPTH_WINDOW_END"] = (
        data["DEPTH_WINDOW_START"] + WINDOW_SIZE
    )

    data["RESIDUAL_RECALCULATED"] = (
        data["PREDICTED_PHIF"] - data["PHIF"]
    )

    grouped = data.groupby(
        ["WELL", "DEPTH_WINDOW_START", "DEPTH_WINDOW_END"],
        sort=True,
        observed=True,
    )

    summary = grouped.agg(
        N_SAMPLES=("PHIF", "size"),
        DEPTH_OBS_MIN=("DEPTH", "min"),
        DEPTH_OBS_MAX=("DEPTH", "max"),
        ACTUAL_PHIF_MEAN=("PHIF", "mean"),
        ACTUAL_PHIF_MEDIAN=("PHIF", "median"),
        ACTUAL_PHIF_STD=("PHIF", "std"),
        PREDICTED_PHIF_MEAN=("PREDICTED_PHIF", "mean"),
        PREDICTED_PHIF_MEDIAN=("PREDICTED_PHIF", "median"),
        PREDICTED_PHIF_STD=("PREDICTED_PHIF", "std"),
        MEAN_BIAS=("RESIDUAL_RECALCULATED", "mean"),
        MAE=("ABS_ERROR", "mean"),
    ).reset_index()

    rmse = (
        grouped["RESIDUAL_RECALCULATED"]
        .apply(lambda x: np.sqrt(np.mean(x ** 2)))
        .rename("RMSE")
        .reset_index()
    )

    summary = summary.merge(
        rmse,
        on=[
            "WELL",
            "DEPTH_WINDOW_START",
            "DEPTH_WINDOW_END",
        ],
        how="left",
        validate="one_to_one",
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("OOF PHIF RESERVOIR CHARACTERIZATION")
    print("=" * 68)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"OOF prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    validate_data(df)

    print(f"Input rows: {len(df):,}")
    print(f"Input wells: {df['WELL'].nunique()}")
    print(
        f"Depth range: "
        f"{df['DEPTH'].min():.3f} to {df['DEPTH'].max():.3f}"
    )
    print(f"Depth-window width: {WINDOW_SIZE:g} depth units")

    well_summary = summarize_by_well(df)
    depth_summary = summarize_by_depth_window(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    well_summary.to_csv(WELL_OUTPUT, index=False)
    depth_summary.to_csv(DEPTH_OUTPUT, index=False)

    print("\n" + "=" * 68)
    print("WELL-LEVEL CHARACTERIZATION")
    print("=" * 68)
    print(
        well_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 68)
    print("DEPTH-WINDOW SAMPLE")
    print("=" * 68)
    print(
        depth_summary.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n" + "=" * 68)
    print("VALIDATION")
    print("=" * 68)
    print(f"Well summary rows: {len(well_summary)}")
    print(f"Depth-window rows: {len(depth_summary)}")
    print(
        "Wells represented in depth summary:",
        depth_summary["WELL"].nunique(),
    )

    print("\nSaved:")
    print(WELL_OUTPUT)
    print(DEPTH_OUTPUT)


if __name__ == "__main__":
    main()
