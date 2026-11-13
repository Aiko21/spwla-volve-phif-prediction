
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports/models/phif_oof_characterization_by_well.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports/figures"

OUTPUT_PATH = OUTPUT_DIR / "phif_oof_well_comparison.png"



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH).sort_values("WELL")

    required = {
        "WELL",
        "ACTUAL_PHIF_MEAN",
        "PREDICTED_PHIF_MEAN",
        "MEAN_BIAS_PRED_MINUS_ACTUAL",
        "MAE",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


    if df.empty:
        raise ValueError("Characterization dataset is empty.")

    if df["WELL"].isna().any() or df["WELL"].duplicated().any():
        raise ValueError(
           "Missing or duplicate WELL identifiers found."
       )

    numeric_columns = [
        "ACTUAL_PHIF_MEAN",
        "PREDICTED_PHIF_MEAN",
        "MEAN_BIAS_PRED_MINUS_ACTUAL",
        "MAE",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values in characterization metrics."
        )

    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError(
            "Infinite values found in characterization metrics."
        )

    if len(df) != 9:
        raise ValueError(
            f"Expected 9 wells, found {len(df)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y = np.arange(len(df))

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 7)
    )

    # Panel 1: Actual vs predicted mean PHIF
    axes[0].scatter(
        df["ACTUAL_PHIF_MEAN"],
        y,
        label="Actual mean PHIF",
    )
    axes[0].scatter(
        df["PREDICTED_PHIF_MEAN"],
        y,
        label="Predicted mean PHIF",
    )

    for i, row in df.reset_index(drop=True).iterrows():
        axes[0].plot(
            [
                row["ACTUAL_PHIF_MEAN"],
                row["PREDICTED_PHIF_MEAN"],
            ],
            [i, i],
            linewidth=1,
        )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["WELL"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean PHIF")
    axes[0].set_title("Actual vs Predicted Mean")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[0].legend()

    # Panel 2: Mean bias
    bias = df["MEAN_BIAS_PRED_MINUS_ACTUAL"]

    axes[1].barh(
        y,
        bias,
    )
    axes[1].axvline(
        0,
        linestyle="--",
        linewidth=1,
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df["WELL"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Predicted - Actual")
    axes[1].set_title("Mean Prediction Bias")
    axes[1].grid(True, axis="x", alpha=0.25)

    # Panel 3: Mean MAE
    axes[2].barh(
        y,
        df["MAE"],
    )
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(df["WELL"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean MAE")
    axes[2].set_title("Mean Error by Well")
    axes[2].grid(True, axis="x", alpha=0.25)

    fig.suptitle(
        "Cross-Well OOF PHIF Characterization",
        fontsize=14,
    )
    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
