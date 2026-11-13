
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_vsh_outlier_details.csv"

FEATURES = ["GR", "RT"]
TARGET = "VSH"
records = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or folder.name in {
        ".git", ".venv", "__pycache__"
    }:
        continue

    inputs = list(folder.glob("*WLC_PETRO_COMPUTED_INPUT*.LAS"))
    outputs = list(folder.glob("*WLC_PETRO_COMPUTED_OUTPUT*.LAS"))

    for ip in inputs:
        for op in outputs:
            try:
                x = lasio.read(str(ip), engine="normal").df()
                y_las = lasio.read(str(op), engine="normal")
                y = y_las.df()

                x.columns = [str(c).upper() for c in x.columns]
                y.columns = [str(c).upper() for c in y.columns]

                x = x.reset_index()
                y = y.reset_index()

                x = x.rename(columns={x.columns[0]: "DEPTH"})
                y = y.rename(columns={y.columns[0]: "DEPTH"})

                x.columns = [str(c).upper() for c in x.columns]
                y.columns = [str(c).upper() for c in y.columns]

                x["DEPTH_KEY"] = pd.to_numeric(
                    x["DEPTH"], errors="coerce"
                ).round(4)
                y["DEPTH_KEY"] = pd.to_numeric(
                    y["DEPTH"], errors="coerce"
                ).round(4)

                if (
                    x["DEPTH_KEY"].duplicated().any()
                    or y["DEPTH_KEY"].duplicated().any()
                ):
                    raise ValueError("Duplicate depth keys")

                if not all(c in x.columns for c in FEATURES):
                    continue
                if TARGET not in y.columns:
                    continue

                merged = x[
                    ["DEPTH", "DEPTH_KEY"] + FEATURES
                ].merge(
                    y[["DEPTH_KEY", TARGET]],
                    on="DEPTH_KEY",
                    how="inner",
                    validate="one_to_one",
                )

                for col in FEATURES + [TARGET]:
                    merged[col] = pd.to_numeric(
                        merged[col], errors="coerce"
                    )

                values = merged[TARGET].to_numpy(dtype=float)
                feature_ok = np.isfinite(
                    merged[FEATURES].to_numpy(dtype=float)
                ).all(axis=1)

                mask = (
                    feature_ok
                    & np.isfinite(values)
                    & (values > 1)
                )

                outliers = merged.loc[mask].copy()

                for _, row in outliers.iterrows():
                    records.append({
                        "well": folder.name,
                        "depth": row["DEPTH"],
                        "GR": row["GR"],
                        "RT": row["RT"],
                        "VSH": row[TARGET],
                        "input_file": ip.name,
                        "output_file": op.name,
                    })

            except Exception as exc:
                print(
                    f"ERROR | {folder.name} | "
                    f"{type(exc).__name__}: {exc}"
                )

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nVSH OUTLIER AUDIT")
print("=" * 70)
print(f"Outlier rows with valid GR and RT: {len(result):,}")

if not result.empty:
    print("\nOutliers by well:")
    print(
        result.groupby("well").agg(
            count=("VSH", "size"),
            max_VSH=("VSH", "max"),
            median_VSH=("VSH", "median"),
            min_GR=("GR", "min"),
            max_GR=("GR", "max"),
        ).round(4).to_string()
    )

    print("\nHighest 15 VSH values:")
    print(
        result.nlargest(15, "VSH")[
            ["well", "depth", "GR", "RT", "VSH"]
        ].to_string(index=False)
    )

print(f"\nSaved: {OUTPUT}")
