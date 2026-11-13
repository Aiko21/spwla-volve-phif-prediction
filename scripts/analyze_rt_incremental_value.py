from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "reports"
    / "models"
    / "phif_depth_feature_cv_results.csv"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "data_quality"
    / "rt_incremental_value_by_well.csv"
)


df = pd.read_csv(INPUT_PATH)

gr = df[df["FEATURE_SET"] == "A_GR_ONLY"].copy()
grrt = df[df["FEATURE_SET"] == "B_GR_RT"].copy()

keys = ["FOLD", "HELD_OUT_WELL", "MODEL"]

gr = gr.rename(
    columns={
        "MAE": "GR_MAE",
        "RMSE": "GR_RMSE",
        "R2": "GR_R2",
    }
)

grrt = grrt.rename(
    columns={
        "MAE": "GRRT_MAE",
        "RMSE": "GRRT_RMSE",
        "R2": "GRRT_R2",
    }
)

merged = gr[keys + ["GR_MAE", "GR_RMSE", "GR_R2"]].merge(
    grrt[keys + ["GRRT_MAE", "GRRT_RMSE", "GRRT_R2"]],
    on=keys,
    how="inner",
)

# Positive = improvement
merged["DELTA_MAE"] = (
    merged["GR_MAE"] - merged["GRRT_MAE"]
)

merged["DELTA_RMSE"] = (
    merged["GR_RMSE"] - merged["GRRT_RMSE"]
)

merged["DELTA_R2"] = (
    merged["GRRT_R2"] - merged["GR_R2"]
)

merged["MAE_IMPROVED"] = merged["DELTA_MAE"] > 0
merged["RMSE_IMPROVED"] = merged["DELTA_RMSE"] > 0
merged["R2_IMPROVED"] = merged["DELTA_R2"] > 0


print("=" * 78)
print("RT INCREMENTAL VALUE BY WELL")
print("=" * 78)

print()

print(
    merged[
        [
            "HELD_OUT_WELL",
            "MODEL",
            "GR_MAE",
            "GRRT_MAE",
            "DELTA_MAE",
            "GR_RMSE",
            "GRRT_RMSE",
            "DELTA_RMSE",
            "GR_R2",
            "GRRT_R2",
            "DELTA_R2",
        ]
    ].to_string(index=False)
)


print()
print("=" * 78)
print("IMPROVEMENT COUNTS")
print("=" * 78)

for model in sorted(merged["MODEL"].unique()):

    subset = merged[merged["MODEL"] == model]

    print()
    print(model)

    print(
        f"MAE improved : "
        f"{subset['MAE_IMPROVED'].sum()}/{len(subset)}"
    )

    print(
        f"RMSE improved: "
        f"{subset['RMSE_IMPROVED'].sum()}/{len(subset)}"
    )

    print(
        f"R2 improved  : "
        f"{subset['R2_IMPROVED'].sum()}/{len(subset)}"
    )


print()
print("=" * 78)
print("AVERAGE INCREMENTAL EFFECT")
print("=" * 78)

print(
    merged.groupby("MODEL")[
        [
            "DELTA_MAE",
            "DELTA_RMSE",
            "DELTA_R2",
        ]
    ].mean().to_string()
)


merged.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 78)
print("OUTPUT")
print("=" * 78)

print(OUTPUT_PATH)

print()
print("Done.")