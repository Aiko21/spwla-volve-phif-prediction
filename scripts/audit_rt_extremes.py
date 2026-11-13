from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phif_train.csv"
)

OUTPUT_DIR = (
    ROOT
    / "reports"
    / "data_quality"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("RT EXTREME VALUE AUDIT")
print("=" * 70)

print(f"Rows: {len(df):,}")


# ============================================================
# QUANTILES
# ============================================================

quantiles = (
    df["RT"]
    .quantile(
        [
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            0.999,
            1.00,
        ]
    )
)

print("\n" + "=" * 70)
print("RT QUANTILES")
print("=" * 70)

print(
    quantiles.to_string()
)


# ============================================================
# UNIQUE EXTREME VALUES
# ============================================================

thresholds = [
    100,
    500,
    1000,
    5000,
    10000,
    50000,
    60000,
]

print("\n" + "=" * 70)
print("RT EXTREME COUNTS")
print("=" * 70)

for threshold in thresholds:

    mask = df["RT"] >= threshold

    print(
        f"RT >= {threshold:>6,.0f}: "
        f"{mask.sum():>6,} rows "
        f"({mask.mean() * 100:.3f}%)"
    )


# ============================================================
# MOST COMMON HIGH RT VALUES
# ============================================================

high_rt = df[
    df["RT"] >= 100
].copy()

print("\n" + "=" * 70)
print("MOST COMMON RT VALUES >= 100")
print("=" * 70)

print(
    high_rt["RT"]
    .value_counts()
    .head(20)
    .to_string()
)


# ============================================================
# EXTREME ROWS
# ============================================================

extreme = df[
    df["RT"] >= 5000
].copy()

extreme = extreme.sort_values(
    "RT",
    ascending=False
)

print("\n" + "=" * 70)
print("EXTREME RT ROWS (RT >= 5000)")
print("=" * 70)

if len(extreme) == 0:

    print("No rows found.")

else:

    print(
        extreme[
            [
                "WELL",
                "DEPTH",
                "GR",
                "RT",
                "PHIF",
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# ============================================================
# EXTREMES BY WELL
# ============================================================

well_summary = (
    df.groupby("WELL")["RT"]
    .agg(
        count="count",
        min="min",
        median="median",
        p95=lambda x: x.quantile(0.95),
        p99=lambda x: x.quantile(0.99),
        max="max",
    )
)

print("\n" + "=" * 70)
print("RT DISTRIBUTION BY WELL")
print("=" * 70)

print(
    well_summary
    .round(4)
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

extreme.to_csv(
    OUTPUT_DIR
    / "rt_extreme_rows_train.csv",
    index=False
)

well_summary.to_csv(
    OUTPUT_DIR
    / "rt_distribution_by_well_train.csv"
)

print("\n" + "=" * 70)
print("AUDIT SAVED")
print("=" * 70)

print(
    OUTPUT_DIR
    / "rt_extreme_rows_train.csv"
)

print(
    OUTPUT_DIR
    / "rt_distribution_by_well_train.csv"
)
