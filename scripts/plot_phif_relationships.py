from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

# RT must be positive for log10
df["LOG10_RT"] = np.log10(
    df["RT"]
)

print("=" * 70)
print("PHIF FEATURE RELATIONSHIP PLOTS")
print("=" * 70)

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# PLOT 1 â€” GR vs PHIF
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["GR"],
    df["PHIF"],
    s=5,
    alpha=0.25
)

plt.xlabel("Gamma Ray (GR)")
plt.ylabel("PHIF")
plt.title("GR vs PHIF â€” Training Wells")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "gr_vs_phif_train.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2 â€” LOG10(RT) vs PHIF
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    df["LOG10_RT"],
    df["PHIF"],
    s=5,
    alpha=0.25
)

plt.xlabel("log10(RT)")
plt.ylabel("PHIF")
plt.title("log10(RT) vs PHIF â€” Training Wells")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "log10rt_vs_phif_train.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3 â€” GR vs PHIF BY WELL
# ============================================================

plt.figure(
    figsize=(10, 7)
)

for well, group in df.groupby("WELL"):

    plt.scatter(
        group["GR"],
        group["PHIF"],
        s=5,
        alpha=0.25,
        label=well
    )

plt.xlabel("Gamma Ray (GR)")
plt.ylabel("PHIF")
plt.title("GR vs PHIF by Well")

plt.legend(
    fontsize=8,
    markerscale=3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "gr_vs_phif_by_well_train.png",
    dpi=200
)

plt.close()


# ============================================================
# PLOT 4 â€” LOG10(RT) vs PHIF BY WELL
# ============================================================

plt.figure(
    figsize=(10, 7)
)

for well, group in df.groupby("WELL"):

    plt.scatter(
        group["LOG10_RT"],
        group["PHIF"],
        s=5,
        alpha=0.25,
        label=well
    )

plt.xlabel("log10(RT)")
plt.ylabel("PHIF")
plt.title("log10(RT) vs PHIF by Well")

plt.legend(
    fontsize=8,
    markerscale=3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "log10rt_vs_phif_by_well_train.png",
    dpi=200
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\nFigures created:")

for filename in [
    "gr_vs_phif_train.png",
    "log10rt_vs_phif_train.png",
    "gr_vs_phif_by_well_train.png",
    "log10rt_vs_phif_by_well_train.png",
]:

    print(
        OUTPUT_DIR / filename
    )

print("\nDONE.")
