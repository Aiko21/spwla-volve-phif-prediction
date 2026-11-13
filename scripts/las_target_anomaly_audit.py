
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_target_anomaly_summary.csv"

TARGETS = ["PHIF", "SW", "VSH", "KLOGH"]
FEATURES = ["GR", "RT"]
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
                x_las = lasio.read(str(ip), engine="normal")
                y_las = lasio.read(str(op), engine="normal")

                x = x_las.df()
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

                available = [t for t in TARGETS if t in y.columns]
                if not available:
                    continue

                merged = x[
                    ["DEPTH_KEY"] + FEATURES
                ].merge(
                    y[["DEPTH_KEY"] + available],
                    on="DEPTH_KEY",
                    how="inner",
                    validate="one_to_one",
                )

                feature_values = merged[FEATURES].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=float)

                feature_ok = np.isfinite(feature_values).all(axis=1)

                for target in available:
                    values = pd.to_numeric(
                        merged[target], errors="coerce"
                    ).to_numpy(dtype=float)

                    valid = feature_ok & np.isfinite(values)
                    v = values[valid]

                    if len(v) == 0:
                        continue

                    if target == "PHIF":
                        flag = np.isclose(v, 0.001)
                        anomaly = np.zeros(len(v), dtype=bool)
                        anomaly |= (v < 0) | (v > 1)

                    elif target == "SW":
                        flag = np.isclose(v, 1.0)
                        anomaly = (v < 0) | (v > 1)

                    elif target == "VSH":
                        flag = v > 1
                        anomaly = (v < 0) | (v > 1)

                    else:  # KLOGH
                        flag = np.isclose(v, 0.001)
                        anomaly = v < 0

                    records.append({
                        "well": folder.name,
                        "target": target,
                        "valid_rows": len(v),
                        "flagged_rows": int(flag.sum()),
                        "flagged_pct": round(
                            100 * flag.mean(), 3
                        ),
                        "out_of_expected_range_rows": int(
                            anomaly.sum()
                        ),
                        "out_of_range_pct": round(
                            100 * anomaly.mean(), 3
                        ),
                    })

            except Exception as exc:
                print(
                    f"ERROR | {folder.name} | "
                    f"{type(exc).__name__}: {exc}"
                )

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nANOMALY AUDIT BY TARGET AND WELL")
print("=" * 75)

if not result.empty:
    print(result.to_string(index=False))

    print("\nOverall summary by target:")
    summary = result.groupby("target").agg(
        wells=("well", "nunique"),
        valid_rows=("valid_rows", "sum"),
        flagged_rows=("flagged_rows", "sum"),
        out_of_range_rows=("out_of_expected_range_rows", "sum"),
    )

    summary["flagged_pct"] = (
        100 * summary["flagged_rows"] / summary["valid_rows"]
    ).round(3)

    summary["out_of_range_pct"] = (
        100 * summary["out_of_range_rows"] / summary["valid_rows"]
    ).round(3)

    print(summary.to_string())

print(f"\nSaved: {OUTPUT}")
