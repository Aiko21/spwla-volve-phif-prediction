
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "data_quality" / "file_inventory.csv"

records = []

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    # Abaikan virtual environment dan metadata Git
    if any(part in {".git", ".venv", "__pycache__"}
           for part in path.parts):
        continue

    records.append({
        "relative_path": path.relative_to(ROOT).as_posix(),
        "parent_folder": path.parent.relative_to(ROOT).as_posix(),
        "file_name": path.name,
        "extension": path.suffix.lower() or "[no extension]",
        "size_bytes": path.stat().st_size,
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("=" * 65)
print("VOLVE DATASET - FILE INVENTORY AUDIT")
print("=" * 65)
print(f"Root directory : {ROOT}")
print(f"Total files    : {len(df)}")
print(f"Output         : {OUTPUT}")

print("\n--- File counts by extension ---")
print(df["extension"].value_counts().to_string())

print("\n--- File counts by top-level folder ---")
df["top_folder"] = df["relative_path"].apply(
    lambda p: p.split("/")[0] if "/" in p else "[root]"
)
print(df["top_folder"].value_counts().to_string())

print("\n--- Largest 15 files ---")
largest = df.nlargest(15, "size_bytes")[
    ["relative_path", "size_bytes"]
]
print(largest.to_string(index=False))

print("\nAudit selesai.")
