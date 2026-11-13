
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_target_definition_audit.csv"

TARGETS = ["PHIF", "SW", "VSH", "KLOGH"]
records = []

for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if folder.name in {".git", ".venv", "__pycache__"}:
        continue

    for path in folder.glob("*WLC_PETRO_COMPUTED_OUTPUT*.LAS"):
        try:
            las = lasio.read(str(path), engine="normal")

            null_value = None
            try:
                null_value = las.well["NULL"].value
            except (KeyError, AttributeError, TypeError):
                pass

            for curve in las.curves:
                mnemonic = str(curve.mnemonic).upper()

                if mnemonic not in TARGETS:
                    continue

                values = pd.to_numeric(
                    pd.Series(las[curve.mnemonic]),
                    errors="coerce"
                ).to_numpy(dtype=float)

                finite = values[np.isfinite(values)]

                row = {
                    "well": folder.name,
                    "file": path.name,
                    "target": mnemonic,
                    "unit": str(curve.unit).strip(),
                    "description": str(curve.descr).strip(),
                    "header_null": null_value,
                    "finite_count": len(finite),
                }

                if len(finite):
                    row.update({
                        "min": float(np.min(finite)),
                        "p01": float(np.percentile(finite, 1)),
                        "median": float(np.median(finite)),
                        "p99": float(np.percentile(finite, 99)),
                        "max": float(np.max(finite)),
                        "negative_count": int((finite < 0).sum()),
                        "zero_count": int((finite == 0).sum()),
                        "gt_1_count": int((finite > 1).sum()),
                        "lt_0_count": int((finite < 0).sum()),
                        "exact_001_count": int(
                            np.isclose(finite, 0.001).sum()
                        ),
                    })

                records.append(row)

        except Exception as exc:
            print(f"ERROR | {path} | {type(exc).__name__}: {exc}")

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nTARGET DEFINITION AUDIT")
print("=" * 80)

if not result.empty:
    print("\nCurve definitions:")
    print(
        result[
            ["well", "target", "unit", "description", "header_null"]
        ].drop_duplicates().to_string(index=False)
    )

    count_columns = [
        "finite_count", "negative_count", "zero_count",
        "gt_1_count", "exact_001_count"
    ]
    for col in count_columns:
        if col not in result.columns:
            result[col] = np.nan

    print("\nPotential anomalies by target:")
    print(
        result.groupby("target")[count_columns]
        .sum(min_count=1)
        .to_string()
    )

    print("\nExtreme values by target:")
    print(
        result.groupby("target").agg(
            minimum=("min", "min"),
            median_of_well_medians=("median", "median"),
            maximum=("max", "max"),
        ).round(5).to_string()
    )

print(f"\nSaved: {OUTPUT}")
