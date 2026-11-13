from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATTERN = "*WLC_PETRO_COMPUTED_INPUT*.LAS"
OUTPUT_PATTERN = "*WLC_PETRO_COMPUTED_OUTPUT*.LAS"

results = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    input_files = list(folder.glob(INPUT_PATTERN))
    output_files = list(folder.glob(OUTPUT_PATTERN))

    if not input_files or not output_files:
        continue

    try:
        x_las = lasio.read(str(input_files[0]), engine="normal")
        y_las = lasio.read(str(output_files[0]), engine="normal")

        x = x_las.df().reset_index()
        y = y_las.df().reset_index()

        x.columns = [str(c).upper() for c in x.columns]
        y.columns = [str(c).upper() for c in y.columns]

        x = x.rename(columns={x.columns[0]: "DEPTH"})
        y = y.rename(columns={y.columns[0]: "DEPTH"})

        for df in (x, y):
            df["DEPTH_KEY"] = pd.to_numeric(
                df["DEPTH"], errors="coerce"
            ).round(4)

        if "GR" not in x.columns or "VSH" not in y.columns:
            continue

        merged = x[["DEPTH_KEY", "GR"]].merge(
            y[["DEPTH_KEY", "VSH"]],
            on="DEPTH_KEY",
            how="inner",
            validate="one_to_one",
        )

        merged["GR"] = pd.to_numeric(
            merged["GR"], errors="coerce"
        )
        merged["VSH"] = pd.to_numeric(
            merged["VSH"], errors="coerce"
        )

        valid = (
            np.isfinite(merged["GR"])
            & np.isfinite(merged["VSH"])
        )
        merged = merged.loc[valid].copy()

        if merged.empty:
            continue

        # Candidate 1: linear formula without clipping
        merged["formula_linear"] = (
            merged["GR"] - 15
        ) / (250 - 15)

        # Candidate 2: linear formula clipped to 0-1
        merged["formula_clipped"] = merged[
            "formula_linear"
        ].clip(0, 1)

        for formula_name in (
            "formula_linear",
            "formula_clipped",
        ):
            error = (
                merged["VSH"] - merged[formula_name]
            ).abs()

            results.append({
                "well": folder.name,
                "formula": formula_name,
                "n": len(merged),
                "median_abs_error": error.median(),
                "mean_abs_error": error.mean(),
                "max_abs_error": error.max(),
                "fraction_error_below_0.001": (
                    error < 0.001
                ).mean(),
                "fraction_vsh_above_1": (
                    merged["VSH"] > 1
                ).mean(),
            })

    except Exception as exc:
        print(f"SKIP {folder.name}: {exc}")

if not results:
    raise RuntimeError(
        "No matching input/output LAS pairs were analyzed."
    )

summary = pd.DataFrame(results)

print("\nVSH FORMULA COMPARISON BY WELL")
print("=" * 100)
print(
    summary.sort_values(
        ["formula", "median_abs_error"]
    ).to_string(index=False)
)

output_path = ROOT / "vsh_formula_comparison.csv"
summary.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
