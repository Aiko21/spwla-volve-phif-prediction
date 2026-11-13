
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_curve_overlap_audit.csv"

# Kandidat input dan target
features = ["GR", "RT", "RD", "RM", "RHOB", "NPHI", "DT"]
targets = ["PHIF", "SW", "VSH", "KLOGH"]

las_files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".las"
    and not any(
        part in {".git", ".venv", "__pycache__"}
        for part in p.parts
    )
)

records = []

for i, path in enumerate(las_files, start=1):
    rel = path.relative_to(ROOT).as_posix()
    print(f"[{i}/{len(las_files)}] {rel}")

    try:
        las = lasio.read(str(path), engine="normal")
        df = las.df()

        # Normalisasi nama kurva
        curve_map = {str(c).upper(): c for c in df.columns}
        available = set(curve_map)

        feature_present = [
            c for c in features if c in available
        ]
        target_present = [
            c for c in targets if c in available
        ]

        record = {
            "relative_path": rel,
            "file_name": path.name,
            "sample_count": len(df),
            "feature_count": len(feature_present),
            "target_count": len(target_present),
            "feature_curves": ", ".join(feature_present),
            "target_curves": ", ".join(target_present),
            "has_GR_RT": (
                "GR" in available and "RT" in available
            ),
            "has_PHIF_SW": (
                "PHIF" in available and "SW" in available
            ),
            "has_all_four_targets": all(
                c in available for c in targets
            ),
            "read_status": "success",
        }

        # Hitung sampel valid hanya jika GR dan RT tersedia
        # serta masing-masing target tersedia.
        for target in targets:
            key = f"valid_GR_RT_{target}"
            required = ["GR", "RT", target]

            if all(c in available for c in required):
                valid = np.ones(len(df), dtype=bool)

                for mnemonic in required:
                    values = pd.to_numeric(
                        df[curve_map[mnemonic]],
                        errors="coerce",
                    ).to_numpy(dtype=float)

                    valid &= np.isfinite(values)

                record[key] = int(valid.sum())
            else:
                record[key] = np.nan

        records.append(record)

    except Exception as exc:
        records.append({
            "relative_path": rel,
            "file_name": path.name,
            "read_status": f"error: {exc}",
        })

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("CURVE OVERLAP AUDIT")
print("=" * 70)

print("\nJumlah file dengan kombinasi kurva:")
for col in [
    "has_GR_RT",
    "has_PHIF_SW",
    "has_all_four_targets",
]:
    if col in result.columns:
        print(f"{col}: {int(result[col].fillna(False).sum())}")

print("\nRingkasan per file:")
columns = [
    "relative_path",
    "sample_count",
    "feature_curves",
    "target_curves",
    "valid_GR_RT_PHIF",
    "valid_GR_RT_SW",
    "valid_GR_RT_VSH",
    "valid_GR_RT_KLOGH",
]
available_columns = [
    c for c in columns if c in result.columns
]
print(result[available_columns].to_string(index=False))

print(f"\nLaporan tersimpan: {OUTPUT}")
