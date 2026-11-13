
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "las_depth_alignment_audit.csv"


def inspect_las(path):
    las = lasio.read(str(path), engine="normal")
    df = las.df()

    depth = pd.to_numeric(
        pd.Index(df.index), errors="coerce"
    ).to_numpy(dtype=float)

    depth = depth[np.isfinite(depth)]
    depth = np.unique(np.sort(depth))

    if len(depth) == 0:
        raise ValueError("No valid depth samples")

    step = np.diff(depth)
    step = step[step > 0]

    return {
        "path": path,
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "sample_count": len(df),
        "median_depth_step": (
            float(np.median(step)) if len(step) else np.nan
        ),
        "curves": ", ".join(str(c) for c in df.columns),
    }


records = []

for folder in sorted(p for p in ROOT.rglob("*") if p.is_dir()):
    if any(
        part in {".git", ".venv", "__pycache__"}
        for part in folder.parts
    ):
        continue

    input_files = sorted(
        p for p in folder.glob("*")
        if p.is_file()
        and p.suffix.lower() == ".las"
        and "WLC_PETRO_COMPUTED_INPUT" in p.name.upper()
    )

    output_files = sorted(
        p for p in folder.glob("*")
        if p.is_file()
        and p.suffix.lower() == ".las"
        and "WLC_PETRO_COMPUTED_OUTPUT" in p.name.upper()
    )

    for input_path in input_files:
        for output_path in output_files:
            print(f"\nChecking well folder: {folder.name}")

            try:
                inp = inspect_las(input_path)
                out = inspect_las(output_path)

                overlap_min = max(
                    inp["depth_min"], out["depth_min"]
                )
                overlap_max = min(
                    inp["depth_max"], out["depth_max"]
                )

                overlap = max(
                    0.0, overlap_max - overlap_min
                )

                records.append({
                    "well_folder": folder.name,
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "input_samples": inp["sample_count"],
                    "output_samples": out["sample_count"],
                    "input_depth_min": inp["depth_min"],
                    "input_depth_max": inp["depth_max"],
                    "output_depth_min": out["depth_min"],
                    "output_depth_max": out["depth_max"],
                    "overlap_depth_min": (
                        overlap_min if overlap > 0 else np.nan
                    ),
                    "overlap_depth_max": (
                        overlap_max if overlap > 0 else np.nan
                    ),
                    "overlap_depth_range": overlap,
                    "input_median_step": inp["median_depth_step"],
                    "output_median_step": out["median_depth_step"],
                    "depth_ranges_overlap": overlap > 0,
                    "input_curves": inp["curves"],
                    "output_curves": out["curves"],
                    "status": "success",
                })

                print(
                    f"  Input : {inp['depth_min']:.3f} "
                    f"to {inp['depth_max']:.3f}"
                )
                print(
                    f"  Output: {out['depth_min']:.3f} "
                    f"to {out['depth_max']:.3f}"
                )
                print(f"  Overlap range: {overlap:.3f}")

            except Exception as exc:
                records.append({
                    "well_folder": folder.name,
                    "input_file": input_path.name,
                    "output_file": output_path.name,
                    "status": f"error: {exc}",
                })

result = pd.DataFrame(records)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("DEPTH ALIGNMENT AUDIT")
print("=" * 70)

if result.empty:
    print("No matching input/output pairs found.")
else:
    print("\nStatus:")
    print(result["status"].value_counts().to_string())

    successful = result[result["status"] == "success"]

    if not successful.empty:
        print(
            "\nPairs with overlapping depth ranges: "
            f"{int(successful['depth_ranges_overlap'].sum())}"
            f"/{len(successful)}"
        )

        columns = [
            "well_folder",
            "input_samples",
            "output_samples",
            "input_depth_min",
            "input_depth_max",
            "output_depth_min",
            "output_depth_max",
            "input_median_step",
            "output_median_step",
            "depth_ranges_overlap",
        ]

        print(successful[columns].to_string(index=False))

print(f"\nSaved report: {OUTPUT}")
