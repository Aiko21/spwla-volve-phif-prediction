
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_target_readiness.csv"

FEATURES = ["GR", "RT"]
TARGETS = ["PHIF", "SW", "VSH", "KLOGH"]

records = []

for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if folder.name in {".git", ".venv", "__pycache__"}:
        continue

    inputs = [
        p for p in folder.glob("*.LAS")
        if "WLC_PETRO_COMPUTED_INPUT" in p.name.upper()
    ]
    outputs = [
        p for p in folder.glob("*.LAS")
        if "WLC_PETRO_COMPUTED_OUTPUT" in p.name.upper()
    ]

    for ip in inputs:
        for op in outputs:
            try:
                x = lasio.read(str(ip), engine="normal").df()
                y = lasio.read(str(op), engine="normal").df()

                x.columns = [str(c).upper() for c in x.columns]
                y.columns = [str(c).upper() for c in y.columns]

                x = x.reset_index()
                y = y.reset_index()

                x = x.rename(columns={x.columns[0]: "DEPTH"})
                y = y.rename(columns={y.columns[0]: "DEPTH"})

                x.columns = [str(c).upper() for c in x.columns]
                y.columns = [str(c).upper() for c in y.columns]

                x["DEPTH"] = pd.to_numeric(
                    x["DEPTH"], errors="coerce"
                )
                y["DEPTH"] = pd.to_numeric(
                    y["DEPTH"], errors="coerce"
                )

                x = x.dropna(subset=["DEPTH"]).copy()
                y = y.dropna(subset=["DEPTH"]).copy()

                x["DEPTH_KEY"] = x["DEPTH"].round(4)
                y["DEPTH_KEY"] = y["DEPTH"].round(4)

                # Hindari merge jika depth key tidak unik.
                if (
                    x["DEPTH_KEY"].duplicated().any()
                    or y["DEPTH_KEY"].duplicated().any()
                ):
                    raise ValueError("Duplicate depth keys")

                missing_features = [
                    c for c in FEATURES if c not in x.columns
                ]
                missing_targets = [
                    c for c in TARGETS if c not in y.columns
                ]

                row = {
                    "well": folder.name,
                    "input_file": ip.name,
                    "output_file": op.name,
                    "missing_features": ", ".join(missing_features),
                    "missing_targets": ", ".join(missing_targets),
                }

                if missing_features:
                    row["status"] = "missing_common_feature"
                    records.append(row)
                    continue

                merged = x[
                    ["DEPTH_KEY"] + FEATURES
                ].merge(
                    y[["DEPTH_KEY"] + [
                        c for c in TARGETS if c in y.columns
                    ]],
                    on="DEPTH_KEY",
                    how="inner",
                    validate="one_to_one",
                )

                feature_valid = np.isfinite(
                    merged[FEATURES].to_numpy(dtype=float)
                ).all(axis=1)

                row["matched_rows"] = len(merged)
                row["valid_GR_RT_rows"] = int(feature_valid.sum())

                for target in TARGETS:
                    if target not in merged.columns:
                        row[f"{target}_valid_rows"] = 0
                        continue

                    target_values = pd.to_numeric(
                        merged[target], errors="coerce"
                    ).to_numpy(dtype=float)

                    valid_target = np.isfinite(target_values)
                    row[f"{target}_valid_rows"] = int(
                        (feature_valid & valid_target).sum()
                    )

                row["status"] = "success"
                records.append(row)

            except Exception as exc:
                records.append({
                    "well": folder.name,
                    "input_file": ip.name,
                    "output_file": op.name,
                    "status": f"error: {type(exc).__name__}: {exc}",
                })

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nTARGET READINESS AUDIT")
print("=" * 70)

if not result.empty:
    print(result.to_string(index=False))

    successful = result[result["status"] == "success"]

    if not successful.empty:
        print("\nTotal valid rows per target (summed across wells):")
        for target in TARGETS:
            col = f"{target}_valid_rows"
            if col in successful.columns:
                print(f"{target}: {successful[col].sum():,}")

print(f"\nSaved: {OUTPUT}")
