
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_target_distribution.csv"

FEATURES = ["GR", "RT"]
TARGETS = ["PHIF", "SW", "VSH", "KLOGH"]

records = []

for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if folder.name in {".git", ".venv", "__pycache__"}:
        continue

    inputs = list(folder.glob("*WLC_PETRO_COMPUTED_INPUT*.LAS"))
    outputs = list(folder.glob("*WLC_PETRO_COMPUTED_OUTPUT*.LAS"))

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

                available_targets = [
                    c for c in TARGETS if c in y.columns
                ]
                if not available_targets:
                    continue

                merged = x[
                    ["DEPTH_KEY"] + FEATURES
                ].merge(
                    y[["DEPTH_KEY"] + available_targets],
                    on="DEPTH_KEY",
                    how="inner",
                    validate="one_to_one",
                )

                feature_values = merged[FEATURES].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)

                feature_valid = np.isfinite(feature_values).all(axis=1)

                for target in available_targets:
                    values = pd.to_numeric(
                        merged[target], errors="coerce"
                    ).to_numpy(dtype=float)

                    valid = feature_valid & np.isfinite(values)
                    v = values[valid]

                    if len(v) == 0:
                        continue

                    records.append({
                        "well": folder.name,
                        "target": target,
                        "valid_rows": len(v),
                        "unique_values": len(np.unique(v)),
                        "mean": np.mean(v),
                        "std": np.std(v),
                        "min": np.min(v),
                        "p01": np.percentile(v, 1),
                        "p25": np.percentile(v, 25),
                        "median": np.median(v),
                        "p75": np.percentile(v, 75),
                        "p99": np.percentile(v, 99),
                        "max": np.max(v),
                    })

            except Exception as exc:
                print(
                    f"ERROR | {folder.name} | {ip.name} | "
                    f"{type(exc).__name__}: {exc}"
                )

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nTARGET DISTRIBUTION AUDIT")
print("=" * 80)

if not result.empty:
    summary = result.groupby("target").agg(
        wells=("well", "nunique"),
        valid_rows=("valid_rows", "sum"),
        mean_of_well_means=("mean", "mean"),
        minimum=("min", "min"),
        maximum=("max", "max"),
        average_median=("median", "mean"),
    ).round(4)

    print("\nSummary by target:")
    print(summary.to_string())

    print("\nStatistics by well:")
    print(result.round(4).to_string(index=False))

print(f"\nSaved: {OUTPUT}")
