
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
WELL = "15_9-F-1"

folder = ROOT / WELL
input_files = list(folder.glob("*WLC_PETRO_COMPUTED_INPUT*.LAS"))
output_files = list(folder.glob("*WLC_PETRO_COMPUTED_OUTPUT*.LAS"))

if not input_files or not output_files:
    raise FileNotFoundError(
        f"Input/output LAS tidak ditemukan di {folder}"
    )

x_las = lasio.read(str(input_files[0]), engine="normal")
y_las = lasio.read(str(output_files[0]), engine="normal")

x = x_las.df()
y = y_las.df()

x.columns = [str(c).upper() for c in x.columns]
y.columns = [str(c).upper() for c in y.columns]

x = x.reset_index().rename(
    columns={x.reset_index().columns[0]: "DEPTH"}
)
y = y.reset_index().rename(
    columns={y.reset_index().columns[0]: "DEPTH"}
)

x.columns = [str(c).upper() for c in x.columns]
y.columns = [str(c).upper() for c in y.columns]

for df in (x, y):
    df["DEPTH_KEY"] = pd.to_numeric(
        df["DEPTH"], errors="coerce"
    ).round(4)

merged = x[["DEPTH", "DEPTH_KEY", "GR"]].merge(
    y[["DEPTH_KEY", "VSH"]],
    on="DEPTH_KEY",
    how="inner",
    validate="one_to_one",
)

merged["GR"] = pd.to_numeric(merged["GR"], errors="coerce")
merged["VSH"] = pd.to_numeric(merged["VSH"], errors="coerce")

valid = (
    np.isfinite(merged["GR"])
    & np.isfinite(merged["VSH"])
)
merged = merged.loc[valid].copy()

# Rumus hanya diuji sebagai hipotesis berdasarkan deskripsi header.
merged["VSH_formula"] = (merged["GR"] - 15) / (250 - 15)
merged["formula_error"] = (
    merged["VSH"] - merged["VSH_formula"]
).abs()

print("\nVSH FORMULA CHECK â€” 15_9-F-1")
print("=" * 70)
print(f"Matched finite GR/VSH rows: {len(merged):,}")

print("\nOverall formula comparison:")
print(
    merged[["VSH", "VSH_formula", "formula_error"]]
    .describe()
    .round(6)
    .to_string()
)

print("\nFormula error statistics by GR range:")
merged["GR_range"] = pd.cut(
    merged["GR"],
    bins=[-np.inf, 15, 250, np.inf],
    labels=["GR <= 15", "15 < GR <= 250", "GR > 250"],
)

print(
    merged.groupby("GR_range", observed=False)["formula_error"]
    .agg(["count", "median", "mean", "max"])
    .round(6)
    .to_string()
)

print("\nLargest formula discrepancies:")
print(
    merged.nlargest(10, "formula_error")[
        ["DEPTH", "GR", "VSH", "VSH_formula", "formula_error"]
    ].to_string(index=False)
)
