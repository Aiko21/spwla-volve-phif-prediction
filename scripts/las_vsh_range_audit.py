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

        x["DEPTH_KEY"] = pd.to_numeric(
            x["DEPTH"], errors="coerce"
        ).round(4)
        y["DEPTH_KEY"] = pd.to_numeric(
            y["DEPTH"], errors="coerce"
        ).round(4)

        if "VSH" not in y.columns:
            continue

        y["VSH"] = pd.to_numeric(
            y["VSH"], errors="coerce"
        )

        vsh = y["VSH"].to_numpy(dtype=float)
        finite = np.isfinite(vsh)
        valid = vsh[finite]

        if len(valid) == 0:
            continue

        records.append({
            "well": folder.name,
            "valid_VSH": len(valid),
            "VSH_min": np.min(valid),
            "VSH_median": np.median(valid),
            "VSH_max": np.max(valid),
            "count_below_0": np.sum(valid < 0),
            "count_above_1": np.sum(valid > 1),
            "fraction_outside_0_1": np.mean(
                (valid < 0) | (valid > 1)
            ),
            "count_exact_0": np.sum(valid == 0),
            "count_exact_1": np.sum(valid == 1),
        })

    except Exception as exc:
        print(f"SKIP {folder.name}: {exc}")

summary = pd.DataFrame(records)

if summary.empty:
    raise RuntimeError("Tidak ada data VSH yang berhasil diaudit.")

summary = summary.sort_values(
    "fraction_outside_0_1", ascending=False
)

print("\nVSH RANGE AUDIT BY WELL")
print("=" * 100)
print(summary.round(5).to_string(index=False))

out_path = ROOT / "vsh_range_audit.csv"
summary.to_csv(out_path, index=False)

print(f"\nSaved: {out_path}")
