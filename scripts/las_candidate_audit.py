
from pathlib import Path
import pandas as pd
import numpy as np
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_candidate_audit.csv"

candidate_curves = [
    "GR", "RT", "RD", "RM",
    "RHOB", "NPHI", "DT", "DTS",
    "PHIF", "SW", "VSH", "KLOGH",
    "SAND_FLAG", "COAL_FLAG", "CARB_FLAG",
]

records = []

las_files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".las"
    and not any(
        part in {".git", ".venv", "__pycache__"}
        for part in p.parts
    )
)

for i, path in enumerate(las_files, start=1):
    relative_path = path.relative_to(ROOT).as_posix()
    print(f"[{i}/{len(las_files)}] {relative_path}")

    try:
        # Engine normal untuk membaca file LAS wrapped.
        las = lasio.read(str(path), engine="normal")
        df = las.df()

        # Samakan huruf mnemonic agar pencocokan konsisten.
        curve_map = {str(c).upper(): c for c in df.columns}

        record = {
            "relative_path": relative_path,
            "file_name": path.name,
            "sample_count": len(df),
            "read_status": "success",
            "error": None,
        }

        valid_masks = []

        for mnemonic in candidate_curves:
            if mnemonic not in curve_map:
                record[f"{mnemonic}_present"] = False
                record[f"{mnemonic}_missing_pct"] = np.nan
                continue

            column = curve_map[mnemonic]
            values = pd.to_numeric(df[column], errors="coerce")
            valid = values.notna() & np.isfinite(values)

            record[f"{mnemonic}_present"] = True
            record[f"{mnemonic}_missing_pct"] = (
                100 * values.isna().mean()
            )

            if mnemonic in ["GR", "RT", "PHIF", "SW", "VSH"]:
                valid_masks.append(valid)

        # Sampel yang memiliki seluruh kurva inti yang tersedia
        # dalam file ini. Tidak mengisi missing values.
        if valid_masks:
            joint_valid = np.logical_and.reduce(
                [mask.to_numpy() for mask in valid_masks]
            )
            record["joint_valid_samples_core"] = int(
                joint_valid.sum()
            )
        else:
            record["joint_valid_samples_core"] = 0

        records.append(record)

    except Exception as exc:
        records.append({
            "relative_path": relative_path,
            "file_name": path.name,
            "sample_count": np.nan,
            "read_status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 65)
print("CANDIDATE CURVE AUDIT")
print("=" * 65)

print("\nRead status:")
print(result["read_status"].value_counts().to_string())

print("\nCurve availability:")
for curve in candidate_curves:
    col = f"{curve}_present"
    if col in result.columns:
        print(
            f"{curve:12s}: "
            f"{int(result[col].fillna(False).sum())} files"
        )

print("\nFiles with core curves:")
core = [
    f"{c}_present"
    for c in ["GR", "RT", "PHIF", "SW", "VSH"]
]
available_core = [
    c for c in core if c in result.columns
]

if available_core:
    selected = result[
        result[available_core].fillna(False).all(axis=1)
    ]
    print(
        selected[
            [
                "relative_path",
                "sample_count",
                "joint_valid_samples_core",
            ]
        ].to_string(index=False)
    )

print(f"\nSaved report: {OUTPUT}")
