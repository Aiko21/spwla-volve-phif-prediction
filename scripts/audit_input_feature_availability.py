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

CANDIDATE_FEATURES = [
    "GR",
    "RT",
    "SW",
    "VSH",
    "KLOGH",
]

OUTPUT_DIR = REPO_ROOT / "reports" / "data_quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def find_curve(las, curve_name):
    """
    Find a LAS curve case-insensitively.
    Returns the actual curve mnemonic or None.
    """
    target = curve_name.upper()

    for curve in las.curves:
        mnemonic = curve.mnemonic.strip()

        if mnemonic.upper() == target:
            return mnemonic

    return None


def safe_numeric(series):
    """
    Convert curve values to numeric and replace LAS NULL values.
    """
    values = pd.to_numeric(series, errors="coerce")

    # LAS NULL value
    values = values.replace(-999.25, np.nan)

    return values


# ============================================================
# AUDIT
# ============================================================

records = []

print("=" * 80)
print("INPUT FEATURE AVAILABILITY AUDIT")
print("=" * 80)

for well in WELL_NAMES:

    las_path = REPO_ROOT / well / INPUT_FILENAME

    print(f"\nWell: {well}")
    print(f"File: {las_path}")

    if not las_path.exists():
        print("  WARNING: INPUT LAS not found")
        continue

    las = lasio.read(las_path)

    print(f"  Curves available: {len(las.curves)}")
    print(f"  Rows: {len(las.df())}")

    df = las.df()

    for feature in CANDIDATE_FEATURES:

        actual_curve = find_curve(las, feature)

        record = {
            "WELL": well,
            "FEATURE": feature,
            "CURVE_FOUND": actual_curve is not None,
            "ACTUAL_MNEMONIC": actual_curve,
            "ROWS": len(df),
            "VALID_COUNT": 0,
            "MISSING_COUNT": len(df),
            "MISSING_PCT": 100.0,
            "MIN": np.nan,
            "MEDIAN": np.nan,
            "MAX": np.nan,
        }

        if actual_curve is not None:

            values = safe_numeric(df[actual_curve])

            valid = values.dropna()

            record["VALID_COUNT"] = len(valid)
            record["MISSING_COUNT"] = values.isna().sum()
            record["MISSING_PCT"] = (
                record["MISSING_COUNT"] / len(values) * 100
                if len(values) > 0
                else np.nan
            )

            if len(valid) > 0:
                record["MIN"] = valid.min()
                record["MEDIAN"] = valid.median()
                record["MAX"] = valid.max()

        records.append(record)

        status = "FOUND" if actual_curve else "NOT FOUND"

        print(
            f"  {feature:<8} {status:<10}"
            f" valid={record['VALID_COUNT']:<7}"
            f" missing={record['MISSING_PCT']:.2f}%"
        )


# ============================================================
# RESULT DATAFRAME
# ============================================================

audit_df = pd.DataFrame(records)

output_path = OUTPUT_DIR / "input_feature_availability.csv"

audit_df.to_csv(output_path, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

summary = (
    audit_df
    .groupby("FEATURE")
    .agg(
        WELLS_FOUND=("CURVE_FOUND", "sum"),
        TOTAL_WELLS=("WELL", "nunique"),
        TOTAL_VALID=("VALID_COUNT", "sum"),
        TOTAL_ROWS=("ROWS", "sum"),
    )
    .reset_index()
)

summary["AVAILABILITY_PCT"] = (
    summary["WELLS_FOUND"] /
    summary["TOTAL_WELLS"] *
    100
)

summary["VALID_PCT"] = (
    summary["TOTAL_VALID"] /
    summary["TOTAL_ROWS"] *
    100
)

print(summary.to_string(index=False))


# ============================================================
# PER-WELL TABLE
# ============================================================

print("\n" + "=" * 80)
print("PER-WELL FEATURE AVAILABILITY")
print("=" * 80)

pivot = audit_df.pivot(
    index="WELL",
    columns="FEATURE",
    values="CURVE_FOUND"
)

print(pivot.to_string())


print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(f"\nSaved:")
print(output_path)
