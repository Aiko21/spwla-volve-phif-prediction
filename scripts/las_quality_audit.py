
from pathlib import Path
import numpy as np
import pandas as pd
import lasio

ROOT = Path(__file__).resolve().parents[1]
FILE_OUTPUT = ROOT / "las_sample_quality.csv"
CURVE_OUTPUT = ROOT / "las_curve_quality.csv"

las_files = sorted(
    p for p in ROOT.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".las"
    and not any(
        part in {".git", ".venv", "__pycache__"}
        for part in p.parts
    )
)

file_results = []
curve_results = []

for i, path in enumerate(las_files, start=1):
    relative_path = path.relative_to(ROOT).as_posix()
    print(f"[{i}/{len(las_files)}] Auditing {relative_path}")

    try:
        las = lasio.read(str(path))
        df = las.df()

        # Identitas file tetap memakai path lengkap relatif,
        # sebab nama file yang sama dapat muncul di beberapa well.
        depth = pd.to_numeric(
            pd.Series(df.index), errors="coerce"
        )
        valid_depth = depth.dropna()

        file_results.append({
            "relative_path": relative_path,
            "status": "success",
            "sample_count": len(df),
            "curve_count_in_data": len(df.columns),
            "depth_min": valid_depth.min()
                if not valid_depth.empty else np.nan,
            "depth_max": valid_depth.max()
                if not valid_depth.empty else np.nan,
            "duplicate_depth_count": int(
                valid_depth.duplicated().sum()
            ),
            "error": None,
        })

        for mnemonic in df.columns:
            values = pd.to_numeric(
                df[mnemonic], errors="coerce"
            )
            non_null = values.dropna()

            curve_results.append({
                "relative_path": relative_path,
                "mnemonic": str(mnemonic),
                "sample_count": len(values),
                "missing_count": int(values.isna().sum()),
                "missing_pct": (
                    values.isna().mean() * 100
                    if len(values) else np.nan
                ),
                "unique_non_null": int(non_null.nunique()),
                "min_value": non_null.min()
                    if not non_null.empty else np.nan,
                "max_value": non_null.max()
                    if not non_null.empty else np.nan,
            })

    except Exception as exc:
        file_results.append({
            "relative_path": relative_path,
            "status": "error",
            "sample_count": np.nan,
            "curve_count_in_data": np.nan,
            "depth_min": np.nan,
            "depth_max": np.nan,
            "duplicate_depth_count": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        })

files_df = pd.DataFrame(file_results)
curves_df = pd.DataFrame(curve_results)

files_df.to_csv(FILE_OUTPUT, index=False, encoding="utf-8-sig")
curves_df.to_csv(CURVE_OUTPUT, index=False, encoding="utf-8-sig")

print("\n" + "=" * 65)
print("LAS SAMPLE QUALITY SUMMARY")
print("=" * 65)

print("\nFile status:")
print(files_df["status"].value_counts().to_string())

successful = files_df[files_df["status"] == "success"]
print(f"\nFiles processed successfully: {len(successful)}")
print(f"Total samples across files: {successful['sample_count'].sum():,.0f}")

print("\nSamples and depth ranges by file:")
print(
    successful[
        [
            "relative_path",
            "sample_count",
            "depth_min",
            "depth_max",
            "duplicate_depth_count",
        ]
    ].to_string(index=False)
)

if not curves_df.empty:
    print("\nCurves with highest missing percentage:")
    print(
        curves_df.sort_values(
            "missing_pct", ascending=False
        ).head(20)[
            [
                "relative_path",
                "mnemonic",
                "missing_pct",
                "unique_non_null",
            ]
        ].to_string(index=False)
    )

    print("\nCurve quality aggregated by mnemonic:")
    summary = curves_df.groupby("mnemonic").agg(
        files_present=("relative_path", "nunique"),
        mean_missing_pct=("missing_pct", "mean"),
        min_unique_values=("unique_non_null", "min"),
    ).sort_values("files_present", ascending=False)

    print(summary.head(30).to_string())

print(f"\nFile report : {FILE_OUTPUT}")
print(f"Curve report: {CURVE_OUTPUT}")
print("Quality audit finished.")
