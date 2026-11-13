
from pathlib import Path
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CURVES = ROOT / "las_curve_inventory.csv"
OUTPUT_FILES = ROOT / "las_file_audit.csv"

las_files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".las"
    and not any(
        part in {".git", ".venv", "__pycache__"}
        for part in p.parts
    )
)

file_records = []
curve_records = []


def get_well_value(las, mnemonic):
    try:
        item = las.well[mnemonic]
        return item.value
    except (KeyError, IndexError, AttributeError):
        return None


print(f"LAS files found: {len(las_files)}")

for i, path in enumerate(las_files, start=1):
    print(f"[{i}/{len(las_files)}] {path.relative_to(ROOT)}")

    try:
        # Baca metadata dan header saja, bukan seluruh log samples.
        las = lasio.read(str(path), ignore_data=True)

        well_name = get_well_value(las, "WELL")
        uwi = get_well_value(las, "UWI")
        start_depth = get_well_value(las, "STRT")
        stop_depth = get_well_value(las, "STOP")
        step = get_well_value(las, "STEP")
        null_value = get_well_value(las, "NULL")

        file_records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "well_name": well_name,
            "uwi": uwi,
            "start_depth": start_depth,
            "stop_depth": stop_depth,
            "step": step,
            "null_value": null_value,
            "curve_count": len(las.curves),
            "status": "success",
            "error": None,
        })

        for curve in las.curves:
            curve_records.append({
                "relative_path": path.relative_to(ROOT).as_posix(),
                "file_name": path.name,
                "well_name": well_name,
                "mnemonic": curve.mnemonic,
                "unit": curve.unit,
                "description": curve.descr,
            })

    except Exception as exc:
        file_records.append({
            "relative_path": path.relative_to(ROOT).as_posix(),
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })

files_df = pd.DataFrame(file_records)
curves_df = pd.DataFrame(curve_records)

files_df.to_csv(OUTPUT_FILES, index=False, encoding="utf-8-sig")
curves_df.to_csv(OUTPUT_CURVES, index=False, encoding="utf-8-sig")

print("\n" + "=" * 65)
print("LAS HEADER AUDIT SUMMARY")
print("=" * 65)

if not files_df.empty:
    print("\nFile status:")
    print(files_df["status"].value_counts().to_string())

    print("\nCurve count per file:")
    print(
        files_df.loc[
            files_df["status"] == "success",
            ["file_name", "curve_count", "start_depth", "stop_depth"]
        ].to_string(index=False)
    )

if not curves_df.empty:
    print("\nMost common curve mnemonics:")
    print(
        curves_df["mnemonic"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\nDistinct curve units:")
    print(sorted(curves_df["unit"].dropna().unique().tolist()))

print(f"\nFile audit saved : {OUTPUT_FILES}")
print(f"Curve inventory  : {OUTPUT_CURVES}")
print("Audit finished.")
