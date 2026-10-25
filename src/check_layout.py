
from pathlib import Path
import pandas as pd
import importlib.util

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "interim" / "lab_rslt.csv"
SAS_FILE = ROOT / "data" / "raw" / "sasinp" / "lab_rslt.inp"

# Load the layout parser from our conversion script.
spec = importlib.util.spec_from_file_location(
    "convert_lab_results",
    ROOT / "src" / "convert_lab_results.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

fields = module.extract_layout(SAS_FILE)
df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)

print("CSV shape:", df.shape)
print("\nColumn names around the potentially merged fields:")

check_names = [
    "B8D38", "Q1D38", "Q2D38", "Q3D38", "Q4D38",
    "RFI38", "RFIDR", "VLOAD", "RO2VL", "RO2AC",
    "R2SVL", "R2SAC", "RTQ2VL", "RTQ2AC",
    "RCOBVL", "RCOBAC", "LPA", "APOA", "APOB",
]

for name in check_names:
    print(f"{name}: {'PRESENT' if name in df.columns else 'MISSING'}")

print("\nColumn positions from the SAS layout:")
for field in fields:
    if field["name"] in check_names:
        print(
            f"{field['name']:8} "
            f"columns {field['start']:3}-{field['end']:3}"
        )

print("\nMissing marker counts ('.' means coded missing here):")
for name in ["VLOAD", "LEU3N", "LEU2N", "LEU4N"]:
    if name in df.columns:
        print(f"{name}: {(df[name].str.strip() == '.').sum():,}")

print("\nDone.")
