from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phif_ml_dataset_v1.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

# ------------------------------------------------------------
# Fixed well-based split
# ------------------------------------------------------------

TRAIN_WELLS = [
    "15_9-F-1A",
    "15_9-F-1B",
    "15_9-F11 T2",
    "15_9-F11A",
    "15_9-F11B",
    "15_9-F12",
]

VALIDATION_WELLS = [
    "15_9-F15D",
]

TEST_WELLS = [
    "15_9-F-1",
    "15_9-F1C",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATE WELL-BASED DATA SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal rows : {len(df):,}")
print(f"Total wells: {df['WELL'].nunique()}")


# ============================================================
# VALIDATE SPLIT DEFINITIONS
# ============================================================

all_wells = set(df["WELL"].unique())

split_wells = (
    set(TRAIN_WELLS)
    | set(VALIDATION_WELLS)
    | set(TEST_WELLS)
)

# Check unknown wells
unknown_wells = split_wells - all_wells

if unknown_wells:
    raise ValueError(
        f"Unknown wells in split definition: "
        f"{sorted(unknown_wells)}"
    )

# Check missing wells
missing_wells = all_wells - split_wells

if missing_wells:
    raise ValueError(
        f"Wells not assigned to any split: "
        f"{sorted(missing_wells)}"
    )

# Check overlap
train_set = set(TRAIN_WELLS)
val_set = set(VALIDATION_WELLS)
test_set = set(TEST_WELLS)

if train_set & val_set:
    raise ValueError("Train/validation well overlap detected.")

if train_set & test_set:
    raise ValueError("Train/test well overlap detected.")

if val_set & test_set:
    raise ValueError("Validation/test well overlap detected.")


# ============================================================
# CREATE SPLITS
# ============================================================

train_df = df[
    df["WELL"].isin(TRAIN_WELLS)
].copy()

validation_df = df[
    df["WELL"].isin(VALIDATION_WELLS)
].copy()

test_df = df[
    df["WELL"].isin(TEST_WELLS)
].copy()


# ============================================================
# VALIDATE ROW COUNTS
# ============================================================

total_split_rows = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

if total_split_rows != len(df):
    raise ValueError(
        "Split row counts do not equal original dataset."
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_split(name, data):

    print("\n" + "-" * 70)
    print(name)

    print(f"Rows : {len(data):,}")
    print(
        f"Wells: "
        f"{data['WELL'].nunique()}"
    )

    print("\nWells:")
    print(
        data.groupby("WELL")["PHIF"]
        .agg(
            count="count",
            median="median",
            mean="mean",
            min="min",
            max="max",
        )
        .round(4)
        .to_string()
    )

    print("\nPHIF summary:")
    print(
        data["PHIF"]
        .describe()
        .round(4)
    )


summarize_split(
    "TRAIN",
    train_df
)

summarize_split(
    "VALIDATION",
    validation_df
)

summarize_split(
    "TEST",
    test_df
)


# ============================================================
# SAVE DATASETS
# ============================================================

train_path = (
    OUTPUT_DIR
    / "phif_train.csv"
)

validation_path = (
    OUTPUT_DIR
    / "phif_validation.csv"
)

test_path = (
    OUTPUT_DIR
    / "phif_test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

validation_df.to_csv(
    validation_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# SAVE SPLIT METADATA
# ============================================================

split_metadata = pd.DataFrame(
    {
        "WELL": (
            TRAIN_WELLS
            + VALIDATION_WELLS
            + TEST_WELLS
        ),
        "SPLIT": (
            ["train"] * len(TRAIN_WELLS)
            + ["validation"] * len(VALIDATION_WELLS)
            + ["test"] * len(TEST_WELLS)
        ),
    }
)

metadata_path = (
    OUTPUT_DIR
    / "phif_well_split_metadata.csv"
)

split_metadata.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT")
print("=" * 70)

print(
    f"Train      : {len(train_df):,} rows"
)

print(
    f"Validation : {len(validation_df):,} rows"
)

print(
    f"Test       : {len(test_df):,} rows"
)

print(
    f"Total      : {total_split_rows:,} rows"
)

print("\nFiles created:")

print(train_path)
print(validation_path)
print(test_path)
print(metadata_path)

print("\nDONE.")
