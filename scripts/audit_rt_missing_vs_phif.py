from pathlib import Path
import pandas as pd
import numpy as np
import lasio


# ============================================================
# CONFIG
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

WELL_NAMES = [
    "15_9-F-1",
    "15_9-F-1A",
    "15_9-F-1B",
    "15_9-F11 T2",
    "15_9-F11A",
    "15_9-F11B",
    "15_9-F12",
    "15_9-F15D",
    "15_9-F1C",
]

INPUT_FILENAME = "WLC_PETRO_COMPUTED_INPUT_1.LAS"
OUTPUT_FILENAME = "WLC_PETRO_COMPUTED_OUTPUT_1.LAS"

OUTPUT_DIR = REPO_ROOT / "reports" / "data_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_curve(las, curve_name):
    target = curve_name.upper()

    for curve in las.curves:
        mnemonic = curve.mnemonic.strip()

        if mnemonic.upper() == target:
            return mnemonic

    return None


def safe_numeric(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace(-999.25, np.nan)
    return values


# ============================================================
# STORAGE
# ============================================================

well_records = []
overall_records = []


print("=" * 80)
print("RT MISSING VS PHIF SELECTION BIAS AUDIT")
print("=" * 80)


# ============================================================
# PROCESS WELLS
# ============================================================

for well in WELL_NAMES:

    input_path = REPO_ROOT / well / INPUT_FILENAME
    output_path = REPO_ROOT / well / OUTPUT_FILENAME

    print(f"\nWell: {well}")

    if not input_path.exists():
        print("  WARNING: INPUT LAS not found")
        continue

    if not output_path.exists():
        print("  WARNING: OUTPUT LAS not found")
        continue

    input_las = lasio.read(input_path)
    output_las = lasio.read(output_path)

    input_df = input_las.df().reset_index()
    output_df = output_las.df().reset_index()

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    input_depth_col = input_df.columns[0]
    output_depth_col = output_df.columns[0]

    input_df["DEPTH_KEY"] = (
        safe_numeric(input_df[input_depth_col])
        .round(4)
    )

    output_df["DEPTH_KEY"] = (
        safe_numeric(output_df[output_depth_col])
        .round(4)
    )

    # --------------------------------------------------------
    # CURVES
    # --------------------------------------------------------

    gr_curve = find_curve(input_las, "GR")
    rt_curve = find_curve(input_las, "RT")
    phif_curve = find_curve(output_las, "PHIF")

    if gr_curve is None:
        print("  WARNING: GR not found")
        continue

    if rt_curve is None:
        print("  WARNING: RT not found")
        continue

    if phif_curve is None:
        print("  WARNING: PHIF not found")
        continue

    # --------------------------------------------------------
    # SELECT REQUIRED COLUMNS
    # --------------------------------------------------------

    inp = input_df[
        [
            "DEPTH_KEY",
            gr_curve,
            rt_curve,
        ]
    ].copy()

    out = output_df[
        [
            "DEPTH_KEY",
            phif_curve,
        ]
    ].copy()

    inp = inp.rename(
        columns={
            gr_curve: "GR",
            rt_curve: "RT",
        }
    )

    out = out.rename(
        columns={
            phif_curve: "PHIF",
        }
    )

    # --------------------------------------------------------
    # DEPTH ALIGNMENT
    # --------------------------------------------------------

    df = inp.merge(
        out,
        on="DEPTH_KEY",
        how="inner",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # NUMERIC CLEANING
    # --------------------------------------------------------

    df["GR"] = safe_numeric(df["GR"])
    df["RT"] = safe_numeric(df["RT"])
    df["PHIF"] = safe_numeric(df["PHIF"])

    # Only rows where PHIF exists can be used for the
    # selection-bias comparison.
    df = df[df["PHIF"].notna()].copy()

    df["RT_STATUS"] = np.where(
        df["RT"].notna(),
        "RT_VALID",
        "RT_MISSING",
    )

    # --------------------------------------------------------
    # PER-WELL SUMMARY
    # --------------------------------------------------------

    for status, group in df.groupby("RT_STATUS"):

        record = {
            "WELL": well,
            "RT_STATUS": status,
            "ROWS": len(group),
            "GR_VALID_ROWS": group["GR"].notna().sum(),
            "PHIF_MEAN": group["PHIF"].mean(),
            "PHIF_MEDIAN": group["PHIF"].median(),
            "PHIF_STD": group["PHIF"].std(),
            "PHIF_MIN": group["PHIF"].min(),
            "PHIF_MAX": group["PHIF"].max(),
            "GR_MEAN": group["GR"].mean(),
            "GR_MEDIAN": group["GR"].median(),
            "GR_STD": group["GR"].std(),
            "GR_MIN": group["GR"].min(),
            "GR_MAX": group["GR"].max(),
        }

        well_records.append(record)

    # --------------------------------------------------------
    # WELL-LEVEL DIFFERENCE
    # --------------------------------------------------------

    valid_rt = df[df["RT_STATUS"] == "RT_VALID"]
    missing_rt = df[df["RT_STATUS"] == "RT_MISSING"]

    if len(valid_rt) > 0 and len(missing_rt) > 0:

        comparison = {
            "WELL": well,
            "RT_VALID_ROWS": len(valid_rt),
            "RT_MISSING_ROWS": len(missing_rt),
            "RT_MISSING_PCT": (
                len(missing_rt) / len(df) * 100
            ),
            "PHIF_MEAN_VALID_RT": valid_rt["PHIF"].mean(),
            "PHIF_MEAN_MISSING_RT": missing_rt["PHIF"].mean(),
            "PHIF_MEAN_DIFF": (
                missing_rt["PHIF"].mean()
                - valid_rt["PHIF"].mean()
            ),
            "PHIF_MEDIAN_VALID_RT": valid_rt["PHIF"].median(),
            "PHIF_MEDIAN_MISSING_RT": missing_rt["PHIF"].median(),
            "PHIF_MEDIAN_DIFF": (
                missing_rt["PHIF"].median()
                - valid_rt["PHIF"].median()
            ),
            "GR_MEAN_VALID_RT": valid_rt["GR"].mean(),
            "GR_MEAN_MISSING_RT": missing_rt["GR"].mean(),
            "GR_MEAN_DIFF": (
                missing_rt["GR"].mean()
                - valid_rt["GR"].mean()
            ),
        }

        overall_records.append(comparison)

        print(
            f"  RT valid:   {len(valid_rt):>7}"
            f" | PHIF mean={valid_rt['PHIF'].mean():.5f}"
        )

        print(
            f"  RT missing: {len(missing_rt):>7}"
            f" | PHIF mean={missing_rt['PHIF'].mean():.5f}"
        )

        print(
            f"  PHIF mean difference:"
            f" {comparison['PHIF_MEAN_DIFF']:+.5f}"
        )

    elif len(valid_rt) > 0:

        print(
            f"  RT valid only: {len(valid_rt)} rows"
        )

    elif len(missing_rt) > 0:

        print(
            f"  RT missing only: {len(missing_rt)} rows"
        )


# ============================================================
# DATAFRAMES
# ============================================================

summary_df = pd.DataFrame(well_records)
comparison_df = pd.DataFrame(overall_records)


# ============================================================
# SAVE
# ============================================================

summary_path = (
    OUTPUT_DIR /
    "rt_missing_vs_phif_summary.csv"
)

comparison_path = (
    OUTPUT_DIR /
    "rt_missing_vs_phif_comparison.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("RT VALID VS RT MISSING â€” PHIF COMPARISON")
print("=" * 80)

if not comparison_df.empty:

    display_cols = [
        "WELL",
        "RT_VALID_ROWS",
        "RT_MISSING_ROWS",
        "RT_MISSING_PCT",
        "PHIF_MEAN_VALID_RT",
        "PHIF_MEAN_MISSING_RT",
        "PHIF_MEAN_DIFF",
        "PHIF_MEDIAN_VALID_RT",
        "PHIF_MEDIAN_MISSING_RT",
        "PHIF_MEDIAN_DIFF",
    ]

    print(
        comparison_df[
            display_cols
        ].to_string(index=False)
    )


# ============================================================
# OVERALL POOLED COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("POOLED COMPARISON")
print("=" * 80)

if not summary_df.empty:

    pooled = (
        summary_df
        .groupby("RT_STATUS")
        .agg(
            ROWS=("ROWS", "sum"),
            PHIF_MEAN=("PHIF_MEAN", "mean"),
            PHIF_MEDIAN=("PHIF_MEDIAN", "mean"),
            GR_MEAN=("GR_MEAN", "mean"),
        )
        .reset_index()
    )

    print(
        pooled.to_string(index=False)
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print("\nSaved:")
print(summary_path)
print(comparison_path)
