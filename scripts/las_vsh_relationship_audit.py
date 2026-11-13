from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATTERN = "*WLC_PETRO_COMPUTED_INPUT*.LAS"
OUTPUT_PATTERN = "*WLC_PETRO_COMPUTED_OUTPUT*.LAS"

all_data = []

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

        merged["well"] = folder.name
        all_data.append(merged[["well", "GR", "VSH"]])

    except Exception as exc:
        print(f"SKIP {folder.name}: {exc}")

if not all_data:
    raise RuntimeError("No valid GR/VSH pairs found.")

data = pd.concat(all_data, ignore_index=True)

# Ringkasan hubungan GR-VSH per rentang GR
bins = [-np.inf, 15, 50, 100, 150, 200, 250, np.inf]
labels = [
    "GR <= 15",
    "15-50",
    "50-100",
    "100-150",
    "150-200",
    "200-250",
    "GR > 250",
]

data["GR_range"] = pd.cut(
    data["GR"],
    bins=bins,
    labels=labels,
    right=True,
)

summary = (
    data.groupby("GR_range", observed=False)
    .agg(
        n=("VSH", "size"),
        GR_median=("GR", "median"),
        VSH_median=("VSH", "median"),
        VSH_mean=("VSH", "mean"),
        VSH_min=("VSH", "min"),
        VSH_max=("VSH", "max"),
        fraction_VSH_above_1=(
            "VSH", lambda s: (s > 1).mean()
        ),
    )
)

print("\nVSH RELATIONSHIP BY GR RANGE â€” ALL WELLS")
print("=" * 100)
print(summary.round(4).to_string())

# Fokus pada sumur dengan anomali VSH
focus = data[data["well"] == "15_9-F-1"].copy()

focus_summary = (
    focus.groupby("GR_range", observed=False)
    .agg(
        n=("VSH", "size"),
        GR_median=("GR", "median"),
        VSH_median=("VSH", "median"),
        VSH_mean=("VSH", "mean"),
        VSH_min=("VSH", "min"),
        VSH_max=("VSH", "max"),
    )
)

print("\nFOCUS WELL: 15_9-F-1")
print("=" * 100)
print(focus_summary.round(4).to_string())

out_path = ROOT / "vsh_relationship_audit.csv"
summary.to_csv(out_path)

print(f"\nSaved: {out_path}")
