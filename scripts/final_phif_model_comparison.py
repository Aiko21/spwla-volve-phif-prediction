from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "phif_hgb_cv_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "models"
    / "final_phif_model_comparison.csv"
)

REQUIRED_COLUMNS = {
    "HELD_OUT_WELL",
    "MODEL",
    "MAE",
    "RMSE",
    "R2",
}

EXPECTED_MODELS = {
    "LINEAR",
    "XGBOOST",
    "HGB",
}


# ============================================================
# LOAD AND VALIDATE RESULTS
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

missing_columns = REQUIRED_COLUMNS - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

df = df[
    [
        "HELD_OUT_WELL",
        "MODEL",
        "MAE",
        "RMSE",
        "R2",
    ]
].copy()

df["MODEL"] = (
    df["MODEL"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["HELD_OUT_WELL"] = (
    df["HELD_OUT_WELL"]
    .astype(str)
    .str.strip()
)

for column in ["MAE", "RMSE", "R2"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

if df.isna().any().any():
    raise ValueError(
        "Input contains missing or invalid values."
    )

if not set(df["MODEL"].unique()).issubset(EXPECTED_MODELS):
    raise ValueError(
        "Unexpected model names found: "
        f"{sorted(set(df['MODEL']) - EXPECTED_MODELS)}"
    )

if set(df["MODEL"].unique()) != EXPECTED_MODELS:
    raise ValueError(
        "Not all expected models are present."
    )

if df.duplicated(
    subset=["MODEL", "HELD_OUT_WELL"]
).any():
    raise ValueError(
        "Duplicate MODEL and HELD_OUT_WELL combinations found."
    )

well_sets = (
    df.groupby("MODEL")["HELD_OUT_WELL"]
    .apply(set)
)

reference_wells = well_sets.iloc[0]

for model_name, wells in well_sets.items():
    if wells != reference_wells:
        raise ValueError(
            f"Inconsistent evaluation wells for {model_name}."
        )

if len(reference_wells) != 9:
    raise ValueError(
        f"Expected 9 evaluated wells, found {len(reference_wells)}."
    )

print("=" * 100)
print("INPUT VALIDATION")
print("=" * 100)
print(f"Source: {INPUT_FILE}")
print(f"Models: {sorted(df['MODEL'].unique())}")
print(f"Evaluated wells: {len(reference_wells)}")
print(f"Total model evaluations: {len(df)}")


# ============================================================
# SUMMARY
# ============================================================

summary = (
    df.groupby("MODEL")
    .agg(
        N_WELLS=("HELD_OUT_WELL", "nunique"),

        MEAN_MAE=("MAE", "mean"),
        MEDIAN_MAE=("MAE", "median"),
        STD_MAE=("MAE", "std"),
        WORST_MAE=("MAE", "max"),

        MEAN_RMSE=("RMSE", "mean"),
        MEDIAN_RMSE=("RMSE", "median"),
        STD_RMSE=("RMSE", "std"),
        WORST_RMSE=("RMSE", "max"),

        MEAN_R2=("R2", "mean"),
        MEDIAN_R2=("R2", "median"),
        STD_R2=("R2", "std"),
        MIN_R2=("R2", "min"),
        MAX_R2=("R2", "max"),

        N_POSITIVE_R2=("R2", lambda x: (x > 0).sum()),
        N_NEGATIVE_R2=("R2", lambda x: (x < 0).sum()),
    )
    .reset_index()
)

summary["MODEL"] = summary["MODEL"].map({
    "LINEAR": "Linear",
    "XGBOOST": "XGBoost",
    "HGB": "HGB",
})

summary = summary.sort_values(
    "MEAN_MAE"
).reset_index(drop=True)


# ============================================================
# WELL-BY-WELL COMPARISON
# ============================================================

pivot = df.pivot(
    index="HELD_OUT_WELL",
    columns="MODEL",
    values=["MAE", "RMSE", "R2"],
)

pivot.columns = [
    f"{metric}_{model.lower()}"
    for metric, model in pivot.columns
]

pivot = pivot.reset_index().rename(
    columns={"HELD_OUT_WELL": "WELL"}
)


# ============================================================
# MODEL WIN COUNTS
# ============================================================

print("\n--- WELL-LEVEL WIN COUNTS ---")

for metric, direction in [
    ("MAE", "lower"),
    ("RMSE", "lower"),
    ("R2", "higher"),
]:

    metric_pivot = df.pivot(
        index="HELD_OUT_WELL",
        columns="MODEL",
        values=metric,
    )

    if direction == "lower":
        winners = metric_pivot.idxmin(axis=1)
    else:
        winners = metric_pivot.idxmax(axis=1)

    print(f"\n{metric}:")

    print(
        winners.value_counts()
        .to_string()
    )


# ============================================================
# HGB VS OTHER MODELS
# ============================================================

print("\n--- HGB IMPROVEMENT ---")

hgb = summary[
    summary["MODEL"] == "HGB"
].iloc[0]

for model_name in ["Linear", "XGBoost"]:

    other = summary[
        summary["MODEL"] == model_name
    ].iloc[0]

    print(f"\nHGB vs {model_name}")

    print(
        "Delta Mean MAE:",
        round(other["MEAN_MAE"] - hgb["MEAN_MAE"], 6),
    )

    print(
        "Delta Mean RMSE:",
        round(other["MEAN_RMSE"] - hgb["MEAN_RMSE"], 6),
    )

    print(
        "Delta Mean R2:",
        round(hgb["MEAN_R2"] - other["MEAN_R2"], 6),
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n--- OVERALL MODEL PERFORMANCE ---")
print(summary.to_string(index=False))

print("\n--- WELL-BY-WELL PERFORMANCE ---")
print(pivot.to_string(index=False))


# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:")
print(OUTPUT_FILE)
