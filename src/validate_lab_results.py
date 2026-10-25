
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "interim" / "lab_rslt.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)

print("=" * 55)
print("MACS LAB RESULTS — INITIAL DATA VALIDATION")
print("=" * 55)

# 1. Basic dataset information
print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Unique participants: {df['CASEID'].nunique():,}")
print(f"Unique visit codes: {df['VISIT'].nunique():,}")

# 2. Check required columns
required = ["CASEID", "VISIT", "LDATY", "LEU3N", "VLOAD"]
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    print("\nMISSING REQUIRED COLUMNS:", missing_columns)
else:
    print("\nAll expected columns are present.")

# 3. Empty values
print("\nBlank values in important columns:")
for col in required:
    if col in df.columns:
        blanks = df[col].astype(str).str.strip().eq("").sum()
        print(f"  {col}: {blanks:,} ({blanks / len(df) * 100:.2f}%)")

# 4. Exact duplicate records
print(f"\nExact duplicate rows: {df.duplicated().sum():,}")

# Multiple records for one participant and visit may be legitimate.
if {"CASEID", "VISIT"}.issubset(df.columns):
    pairs = df[["CASEID", "VISIT"]]
    print(
        "Rows belonging to participant-visit pairs occurring "
        f"more than once: {pairs.duplicated(keep=False).sum():,}"
    )

# 5. Inspect raw value codes; do not interpret them yet.
for col in ["VLOAD", "LEU3N", "VISIT", "LDATY"]:
    if col in df.columns:
        print(f"\nMost common raw values in {col}:")
        print(df[col].replace("", "[blank]").value_counts().head(15).to_string())

# 6. Quick structural checks
print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nLast five records:")
print(df.tail().to_string(index=False))

print("\nValidation script finished.")
