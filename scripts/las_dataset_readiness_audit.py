
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_dataset_readiness.csv"

FEATURES = ["GR", "RT", "RD", "RM", "RHOB", "NPHI", "DT"]
TARGETS = ["PHIF", "SW", "VSH", "KLOGH"]

records = []

for folder in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    if folder.name in {".git", ".venv", "__pycache__"}:
        continue

    input_files = list(folder.glob("*INPUT*.LAS"))
    output_files = list(folder.glob("*OUTPUT*.LAS"))

    # Abaikan folder yang bukan pasangan WLC input/output.
    input_files = [
        p for p in input_files
        if "WLC_PETRO_COMPUTED_INPUT" in p.name.upper()
    ]
    output_files = [
        p for p in output_files
        if "WLC_PETRO_COMPUTED_OUTPUT" in p.name.upper()
    ]

    if not input_files or not output_files:
        continue

    for input_path in input_files:
        for output_path in output_files:
            print(f"Auditing {folder.name}")

            try:
                input_las = lasio.read(
                    str(input_path), engine="normal"
                )
                output_las = lasio.read(
                    str(output_path), engine="normal"
                )

                x = input_las.df().reset_index()
                y = output_las.df().reset_index()

                x = x.rename(columns={x.columns[0]: "DEPTH"})
                y = y.rename(columns={y.columns[0]: "DEPTH"})

                # Normalisasi nama kolom.
                x.columns = [
                    str(c).upper() for c in x.columns
                ]
                y.columns = [
                    str(c).upper() for c in y.columns
                ]

                x["DEPTH"] = pd.to_numeric(
                    x["DEPTH"], errors="coerce"
                )
                y["DEPTH"] = pd.to_numeric(
                    y["DEPTH"], errors="coerce"
                )

                x = x.dropna(subset=["DEPTH"]).copy()
                y = y.dropna(subset=["DEPTH"]).copy()

                # Samakan representasi desimal untuk menghindari
                # perbedaan floating-point yang sangat kecil.
                x["DEPTH_KEY"] = x["DEPTH"].round(4)
                y["DEPTH_KEY"] = y["DEPTH"].round(4)

                duplicate_x = int(
                    x["DEPTH_KEY"].duplicated().sum()
                )
                duplicate_y = int(
                    y["DEPTH_KEY"].duplicated().sum()
                )

                record = {
                    "well": folder.name,
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "input_rows": len(x),
                    "output_rows": len(y),
                    "duplicate_input_depth_keys": duplicate_x,
                    "duplicate_output_depth_keys": duplicate_y,
                }

                # Jangan melakukan merge ambigu jika kedalaman
                # menjadi duplikat setelah pembulatan.
                if duplicate_x or duplicate_y:
                    record["status"] = "duplicate_depth_key"
                    records.append(record)
                    print("  SKIP: duplicate rounded depth keys")
                    continue

                available_features = [
                    c for c in FEATURES if c in x.columns
                ]
                available_targets = [
                    c for c in TARGETS if c in y.columns
                ]

                record["available_features"] = ", ".join(
                    available_features
                )
                record["available_targets"] = ", ".join(
                    available_targets
                )

                if not available_features or not available_targets:
                    record["status"] = "missing_required_curves"
                    records.append(record)
                    continue

                # Konversi fitur dan target ke numerik.
                for c in available_features:
                    x[c] = pd.to_numeric(x[c], errors="coerce")
                    x[c] = x[c].replace(
                        [np.inf, -np.inf], np.nan
                    )

                for c in available_targets:
                    y[c] = pd.to_numeric(y[c], errors="coerce")
                    y[c] = y[c].replace(
                        [np.inf, -np.inf], np.nan
                    )

                # Pertahankan hanya fitur dan target yang tersedia.
                x = x[
                    ["DEPTH_KEY", "DEPTH"] + available_features
                ]
                y = y[
                    ["DEPTH_KEY", "DEPTH"] + available_targets
                ]

                merged = x.merge(
                    y,
                    on="DEPTH_KEY",
                    how="inner",
                    suffixes=("_INPUT", "_OUTPUT"),
                    validate="one_to_one",
                )

                feature_complete = merged[
                    available_features
                ].notna().all(axis=1)

                target_complete = merged[
                    available_targets
                ].notna().all(axis=1)

                record["matched_depth_rows"] = len(merged)
                record["feature_complete_rows"] = int(
                    feature_complete.sum()
                )
                record["target_complete_rows"] = int(
                    target_complete.sum()
                )
                record["complete_feature_target_rows"] = int(
                    (feature_complete & target_complete).sum()
                )
                record["depth_difference_max"] = float(
                    (
                        merged["DEPTH_INPUT"]
                        - merged["DEPTH_OUTPUT"]
                    ).abs().max()
                ) if len(merged) else np.nan

                record["status"] = "success"
                records.append(record)

                print(
                    f"  Matched depths: {len(merged):,}"
                )
                print(
                    "  Complete feature + target rows: "
                    f"{record['complete_feature_target_rows']:,}"
                )

            except Exception as exc:
                records.append({
                    "well": folder.name,
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "status": f"error: {type(exc).__name__}: {exc}",
                })
                print(f"  ERROR: {exc}")

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("DATASET READINESS AUDIT")
print("=" * 70)

if not result.empty:
    print("\nStatus:")
    print(result["status"].value_counts().to_string())

    cols = [
        "well",
        "input_rows",
        "output_rows",
        "matched_depth_rows",
        "feature_complete_rows",
        "target_complete_rows",
        "complete_feature_target_rows",
        "depth_difference_max",
        "status",
    ]
    cols = [c for c in cols if c in result.columns]
    print("\nPer-well summary:")
    print(result[cols].to_string(index=False))

print(f"\nSaved: {OUTPUT}")
