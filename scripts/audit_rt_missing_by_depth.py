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

OUTPUT_DIR = REPO_ROOT / "reports" / "data_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

DEPTH_BIN_SIZE = 100.0


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
# AUDIT STORAGE
# ============================================================

depth_records = []
interval_records = []


print("=" * 80)
print("RT MISSINGNESS BY DEPTH AUDIT")
print("=" * 80)


# ============================================================
# PROCESS EACH WELL
# ============================================================

for well in WELL_NAMES:

    las_path = REPO_ROOT / well / INPUT_FILENAME

    print(f"\nWell: {well}")
    print(f"File: {las_path}")

    if not las_path.exists():
        print("  WARNING: INPUT LAS not found")
        continue

    las = lasio.read(las_path)

    df = las.df().reset_index()

    # --------------------------------------------------------
    # DEPTH
    # --------------------------------------------------------

    depth_col = df.columns[0]

    depth = safe_numeric(df[depth_col])

    # --------------------------------------------------------
    # RT
    # --------------------------------------------------------

    rt_curve = find_curve(las, "RT")

    if rt_curve is None:
        print("  WARNING: RT curve not found")
        continue

    rt = safe_numeric(df[rt_curve])

    audit = pd.DataFrame({
        "DEPTH": depth,
        "RT": rt,
    })

    audit = audit.dropna(subset=["DEPTH"]).copy()

    audit["RT_MISSING"] = audit["RT"].isna()

    # --------------------------------------------------------
    # BASIC SUMMARY
    # --------------------------------------------------------

    total_rows = len(audit)
    missing_rows = audit["RT_MISSING"].sum()

    missing_pct = (
        missing_rows / total_rows * 100
        if total_rows > 0
        else np.nan
    )

    print(f"  Total rows:       {total_rows}")
    print(f"  Missing RT rows:  {missing_rows}")
    print(f"  Missing RT %:      {missing_pct:.2f}%")

    # --------------------------------------------------------
    # DEPTH BINS
    # --------------------------------------------------------

    depth_min = np.floor(audit["DEPTH"].min() / DEPTH_BIN_SIZE) * DEPTH_BIN_SIZE

    depth_max = np.ceil(audit["DEPTH"].max() / DEPTH_BIN_SIZE) * DEPTH_BIN_SIZE

    bins = np.arange(
        depth_min,
        depth_max + DEPTH_BIN_SIZE,
        DEPTH_BIN_SIZE
    )

    if len(bins) < 2:
        continue

    audit["DEPTH_BIN_START"] = pd.cut(
        audit["DEPTH"],
        bins=bins,
        right=False,
        include_lowest=True,
        labels=bins[:-1]
    )

    grouped = (
        audit
        .groupby("DEPTH_BIN_START", observed=False)
        .agg(
            TOTAL_ROWS=("RT_MISSING", "size"),
            MISSING_RT_ROWS=("RT_MISSING", "sum"),
        )
        .reset_index()
    )

    grouped["WELL"] = well

    grouped["DEPTH_BIN_START"] = pd.to_numeric(
        grouped["DEPTH_BIN_START"],
        errors="coerce"
    )

    grouped["DEPTH_BIN_END"] = (
        grouped["DEPTH_BIN_START"] + DEPTH_BIN_SIZE
    )

    grouped["MISSING_RT_PCT"] = (
        grouped["MISSING_RT_ROWS"] /
        grouped["TOTAL_ROWS"] *
        100
    )

    grouped = grouped[
        grouped["TOTAL_ROWS"] > 0
    ].copy()

    depth_records.append(grouped)

    # --------------------------------------------------------
    # CONTIGUOUS MISSING INTERVALS
    # --------------------------------------------------------

    audit = audit.sort_values("DEPTH").reset_index(drop=True)

    audit["GROUP_CHANGE"] = (
        audit["RT_MISSING"] != audit["RT_MISSING"].shift()
    ).cumsum()

    missing_groups = (
        audit[audit["RT_MISSING"]]
        .groupby("GROUP_CHANGE")
        .agg(
            START_DEPTH=("DEPTH", "min"),
            END_DEPTH=("DEPTH", "max"),
            ROWS=("DEPTH", "size"),
        )
        .reset_index(drop=True)
    )

    if len(missing_groups) > 0:

        missing_groups["INTERVAL_LENGTH"] = (
            missing_groups["END_DEPTH"] -
            missing_groups["START_DEPTH"]
        )

        missing_groups["WELL"] = well

        interval_records.append(missing_groups)


# ============================================================
# SAVE DEPTH-BIN RESULTS
# ============================================================

if depth_records:

    depth_df = pd.concat(
        depth_records,
        ignore_index=True
    )

    depth_output = (
        OUTPUT_DIR /
        "rt_missing_by_depth_100m.csv"
    )

    depth_df.to_csv(
        depth_output,
        index=False
    )

else:

    depth_df = pd.DataFrame()

    depth_output = (
        OUTPUT_DIR /
        "rt_missing_by_depth_100m.csv"
    )


# ============================================================
# SAVE CONTIGUOUS INTERVAL RESULTS
# ============================================================

if interval_records:

    interval_df = pd.concat(
        interval_records,
        ignore_index=True
    )

    interval_output = (
        OUTPUT_DIR /
        "rt_missing_intervals.csv"
    )

    interval_df.to_csv(
        interval_output,
        index=False
    )

else:

    interval_df = pd.DataFrame()

    interval_output = (
        OUTPUT_DIR /
        "rt_missing_intervals.csv"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MISSING RT BY WELL")
print("=" * 80)

if not depth_df.empty:

    well_summary = (
        depth_df
        .groupby("WELL")
        .agg(
            TOTAL_ROWS=("TOTAL_ROWS", "sum"),
            MISSING_RT_ROWS=("MISSING_RT_ROWS", "sum"),
        )
        .reset_index()
    )

    well_summary["MISSING_RT_PCT"] = (
        well_summary["MISSING_RT_ROWS"] /
        well_summary["TOTAL_ROWS"] *
        100
    )

    print(
        well_summary.to_string(index=False)
    )


# ============================================================
# MOST SEVERE DEPTH INTERVALS
# ============================================================

print("\n" + "=" * 80)
print("DEPTH INTERVALS WITH HIGHEST RT MISSINGNESS")
print("=" * 80)

if not depth_df.empty:

    severe = (
        depth_df[
            depth_df["MISSING_RT_ROWS"] > 0
        ]
        .sort_values(
            "MISSING_RT_PCT",
            ascending=False
        )
        .head(30)
    )

    print(
        severe[
            [
                "WELL",
                "DEPTH_BIN_START",
                "DEPTH_BIN_END",
                "TOTAL_ROWS",
                "MISSING_RT_ROWS",
                "MISSING_RT_PCT",
            ]
        ].to_string(index=False)
    )


# ============================================================
# LONGEST CONTIGUOUS MISSING INTERVALS
# ============================================================

print("\n" + "=" * 80)
print("LONGEST CONTIGUOUS RT MISSING INTERVALS")
print("=" * 80)

if not interval_df.empty:

    longest = (
        interval_df
        .sort_values(
            ["ROWS", "INTERVAL_LENGTH"],
            ascending=False
        )
        .head(30)
    )

    print(
        longest[
            [
                "WELL",
                "START_DEPTH",
                "END_DEPTH",
                "INTERVAL_LENGTH",
                "ROWS",
            ]
        ].to_string(index=False)
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(f"\nSaved:")
print(depth_output)
print(interval_output)
