
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "reports/models/final_phif_oof_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/figures/phif_oof_profiles"
)



def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH)

    required = {
        "WELL",
        "DEPTH",
        "GR",
        "PHIF",
        "PREDICTED_PHIF",
        "RESIDUAL",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for well, group in df.groupby("WELL", sort=True):
        group = group.sort_values("DEPTH")

        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(14, 10),
            sharey=True,
        )

        depth = group["DEPTH"]

        # Panel 1: GR and resistivity
        axes[0].plot(
            group["GR"],
            depth,
            linewidth=0.8,
            label="GR",
        )
        axes[0].set_xlabel("GR")
        axes[0].set_ylabel("Depth")
        axes[0].set_title("Gamma Ray")
        axes[0].grid(True, alpha=0.25)

        # Panel 2: actual and OOF predicted PHIF
        axes[1].plot(
            group["PHIF"],
            depth,
            linewidth=0.8,
            label="Actual PHIF",
        )
        axes[1].plot(
            group["PREDICTED_PHIF"],
            depth,
            linewidth=0.8,
            label="OOF Predicted PHIF",
        )
        axes[1].set_xlabel("PHIF")
        axes[1].set_title("Actual vs Prediction")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best")

        # Panel 3: residual
        axes[2].plot(
            group["RESIDUAL"],
            depth,
            linewidth=0.8,
            label="Predicted - Actual",
        )
        axes[2].axvline(
            0,
            linestyle="--",
            linewidth=0.8,
        )
        axes[2].set_xlabel("Residual")
        axes[2].set_title("OOF Prediction Error")
        axes[2].grid(True, alpha=0.25)

        # Conventional well-log display:
        # depth increases downward.
        axes[0].invert_yaxis()

        fig.suptitle(
            f"{well} — OOF PHIF Characterization",
            fontsize=13,
        )

        fig.tight_layout()

        safe_well = (
            well.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        output_path = OUTPUT_DIR / (
            f"{safe_well}_phif_oof_profile.png"
        )

        fig.savefig(
            output_path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

    print("\nAll well profiles generated.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
