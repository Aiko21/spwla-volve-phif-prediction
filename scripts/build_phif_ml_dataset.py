from pathlib import Path
import numpy as np
import pandas as pd
import lasio


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

WELLS = [
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

FEATURES = ["GR", "RT"]
TARGET = "PHIF"


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_wlc_files(well_name: str, root: Path):
    """
    Locate WLC input/output LAS files inside the well folder.
    """

    well_dir = root / well_name

    if not well_dir.exists():
        raise FileNotFoundError(
            f"Well directory not found: {well_dir}"
        )

    input_path = (
        well_dir
        / "WLC_PETRO_COMPUTED_INPUT_1.LAS"
    )

    output_path = (
        well_dir
        / "WLC_PETRO_COMPUTED_OUTPUT_1.LAS"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"WLC input file not found:\n{input_path}"
        )

    if not output_path.exists():
        raise FileNotFoundError(
            f"WLC output file not found:\n{output_path}"
        )

    return input_path, output_path


# ============================================================
# LAS READER
# ============================================================

def read_las(path: Path) -> pd.DataFrame:

    las = lasio.read(path)

    df = las.df().reset_index()

    depth_col = df.columns[0]

    df = df.rename(
        columns={
            depth_col: "DEPTH"
        }
    )

    df.columns = [
        str(col).strip().upper()
        for col in df.columns
    ]

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BUILD PHIF ML DATASET")
    print("=" * 70)

    all_rows = []
    audit_rows = []

    for well in WELLS:

        print("\n" + "-" * 70)
        print(f"Processing: {well}")

        input_path, output_path = find_wlc_files(
            well,
            ROOT
        )

        print(f"INPUT : {input_path}")
        print(f"OUTPUT: {output_path}")

        # ----------------------------------------------------
        # Read INPUT
        # ----------------------------------------------------

        input_df = read_las(input_path)

        # ----------------------------------------------------
        # Read OUTPUT
        # ----------------------------------------------------

        output_df = read_las(output_path)

        # ----------------------------------------------------
        # Validate required curves
        # ----------------------------------------------------

        missing_features = [
            col
            for col in FEATURES
            if col not in input_df.columns
        ]

        if missing_features:
            raise ValueError(
                f"{well}: missing input curves: "
                f"{missing_features}"
            )

        if TARGET not in output_df.columns:
            raise ValueError(
                f"{well}: missing target curve: {TARGET}"
            )

        # ----------------------------------------------------
        # Keep required columns only
        # ----------------------------------------------------

        input_data = input_df[
            ["DEPTH"] + FEATURES
        ].copy()

        output_data = output_df[
            ["DEPTH", TARGET]
        ].copy()

        # ----------------------------------------------------
        # Convert numeric
        # ----------------------------------------------------

        for col in input_data.columns:
            input_data[col] = pd.to_numeric(
                input_data[col],
                errors="coerce"
            )

        for col in output_data.columns:
            output_data[col] = pd.to_numeric(
                output_data[col],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Depth alignment
        # ----------------------------------------------------

        input_data["DEPTH_KEY"] = (
            input_data["DEPTH"].round(4)
        )

        output_data["DEPTH_KEY"] = (
            output_data["DEPTH"].round(4)
        )

        merged = pd.merge(
            input_data,
            output_data[
                ["DEPTH_KEY", TARGET]
            ],
            on="DEPTH_KEY",
            how="inner",
            validate="one_to_one"
        )

        # Use input depth as canonical depth
        merged["DEPTH"] = merged["DEPTH"]

        rows_matched = len(merged)

        # ----------------------------------------------------
        # Quality flags
        # ----------------------------------------------------

        merged["MISSING_GR"] = merged["GR"].isna()
        merged["MISSING_RT"] = merged["RT"].isna()
        merged["MISSING_PHIF"] = merged["PHIF"].isna()

        merged["VALID_FEATURES"] = (
            merged["GR"].notna()
            & merged["RT"].notna()
        )

        merged["VALID_TARGET"] = (
            merged["PHIF"].notna()
        )

        merged["ML_READY"] = (
            merged["VALID_FEATURES"]
            & merged["VALID_TARGET"]
        )

        # ----------------------------------------------------
        # Add well ID
        # ----------------------------------------------------

        merged.insert(
            0,
            "WELL",
            well
        )

        # ----------------------------------------------------
        # Keep ML-ready rows
        # ----------------------------------------------------

        ml_data = merged.loc[
            merged["ML_READY"]
        ].copy()

        rows_ml_ready = len(ml_data)

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        audit_rows.append(
            {
                "WELL": well,
                "INPUT_FILE": str(input_path),
                "OUTPUT_FILE": str(output_path),
                "INPUT_ROWS": len(input_data),
                "OUTPUT_ROWS": len(output_data),
                "MATCHED_ROWS": rows_matched,
                "VALID_GR": int(
                    merged["GR"].notna().sum()
                ),
                "VALID_RT": int(
                    merged["RT"].notna().sum()
                ),
                "VALID_PHIF": int(
                    merged["PHIF"].notna().sum()
                ),
                "ML_READY_ROWS": rows_ml_ready,
            }
        )

        all_rows.append(ml_data)

        print(f"Input rows       : {len(input_data):,}")
        print(f"Output rows      : {len(output_data):,}")
        print(f"Matched rows     : {rows_matched:,}")
        print(
            f"Valid GR         : "
            f"{merged['GR'].notna().sum():,}"
        )
        print(
            f"Valid RT         : "
            f"{merged['RT'].notna().sum():,}"
        )
        print(
            f"Valid PHIF       : "
            f"{merged['PHIF'].notna().sum():,}"
        )
        print(f"ML-ready rows    : {rows_ml_ready:,}")

    # ========================================================
    # COMBINE DATASET
    # ========================================================

    dataset = pd.concat(
        all_rows,
        ignore_index=True
    )

    audit = pd.DataFrame(
        audit_rows
    )

    # Remove technical merge key
    dataset = dataset.drop(
        columns=["DEPTH_KEY"]
    )

    # Sort deterministically
    dataset = dataset.sort_values(
        ["WELL", "DEPTH"]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Total ML-ready rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Unique wells: "
        f"{dataset['WELL'].nunique()}"
    )

    print("\nMissing values:")
    print(
        dataset[
            ["GR", "RT", "PHIF"]
        ].isna().sum()
    )

    print("\nPHIF statistics:")
    print(
        dataset["PHIF"].describe()
    )

    print("\nRows per well:")
    print(
        dataset.groupby("WELL")
        .size()
        .sort_values(
            ascending=False
        )
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dataset = (
        ROOT
        / "data"
        / "processed"
        / "phif_ml_dataset_v1.csv"
    )

    output_audit = (
        ROOT
        / "reports"
        / "data_quality"
        / "phif_ml_dataset_audit_v1.csv"
    )

    output_dataset.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_audit.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset.to_csv(
        output_dataset,
        index=False
    )

    audit.to_csv(
        output_audit,
        index=False
    )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Dataset : {output_dataset}"
    )

    print(
        f"Audit   : {output_audit}"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
