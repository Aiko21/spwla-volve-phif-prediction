from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATTERN = "*WLC_PETRO_COMPUTED_INPUT*.LAS"
OUTPUT_PATTERN = "*WLC_PETRO_COMPUTED_OUTPUT*.LAS"

records = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir():
        continue

    inputs = list(folder.glob(INPUT_PATTERN))
    outputs = list(folder.glob(OUTPUT_PATTERN))

    if not inputs or not outputs:
        continue

    try:
        x_las = lasio.read(str(inputs[0]), engine="normal")
        y_las = lasio.read(str(outputs[0]), engine="normal")

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

        if not {"GR", "RT"}.issubset(x.columns):
            continue

        if "PHIF" not in y.columns:
            continue

        merged = x[
            ["DEPTH_KEY", "GR", "RT"]
        ].merge(
            y[["DEPTH_KEY", "PHIF"]],
            on="DEPTH_KEY",
            how="inner",
            validate="one_to_one",
        )

        for col in ["GR", "RT", "PHIF"]:
            merged[col] = pd.to_numeric(
                merged[col], errors="coerce"
            )

        valid = np.isfinite(
            merged[["GR", "RT", "PHIF"]]
        ).all(axis=1)

        data = merged.loc[valid].copy()

        if data.empty:
            continue

        phif = data["PHIF"]

        records.append({
            "well": folder.name,
            "matched_rows": len(merged),
            "valid_GR_RT_PHIF": len(data),
            "PHIF_min": phif.min(),
            "PHIF_p01": phif.quantile(0.01),
            "PHIF_p50": phif.median(),
            "PHIF_p99": phif.quantile(0.99),
            "PHIF_max": phif.max(),
            "count_below_0": (phif < 0).sum(),
            "count_above_1": (phif > 1).sum(),
            "count_exact_0": (phif == 0).sum(),
            "count_exact_0_001": (phif == 0.001).sum(),
            "unique_values": phif.nunique(),
        })

    except Exception as exc:
        print(f"SKIP {folder.name}: {exc}")

summary = pd.DataFrame(records)

if summary.empty:
    raise RuntimeError("Tidak ada pasangan GR/RT/PHIF valid.")

summary = summary.sort_values("well")

print("\nPHIF TARGET AUDIT BY WELL")
print("=" * 110)
print(summary.round(5).to_string(index=False))

output_path = ROOT / "phif_target_audit.csv"
summary.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")
