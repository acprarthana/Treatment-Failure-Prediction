
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "interim" / "lab_rslt.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"

# Read every field as text first to protect IDs and visit codes.
df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False
)

# Trim whitespace and convert the source's "." missing marker to NA.
for col in df.columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace(".", pd.NA)

# Convert only fields that are genuinely numeric measurements/dates.
numeric_columns = ["LDATY", "LEU3N", "LEU2N", "LEU4N"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

# Quality checks: stop if identifiers or record structure are damaged.
assert len(df) == 143263, f"Unexpected row count: {len(df)}"
assert df["CASEID"].notna().all(), "Missing participant IDs detected"
assert df["CASEID"].str.fullmatch(r"\d{5}").all(), (
    "CASEID must contain exactly five digits"
)
assert df["VISIT"].notna().all(), "Missing visit codes detected"
assert df["VISIT"].str.fullmatch(r"\d{3}").all(), (
    "VISIT must contain exactly three digits"
)

duplicate_pairs = df.duplicated(["CASEID", "VISIT"]).sum()
assert duplicate_pairs == 0, (
    f"Duplicate participant-visit pairs found: {duplicate_pairs}"
)

# Save the reproducibly cleaned dataset.
df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned dataset shape:", df.shape)
print("Example participant IDs:", df["CASEID"].head(10).tolist())
print("Participant ID lengths:", df["CASEID"].str.len().value_counts().to_dict())
print("Example visit codes:", df["VISIT"].head(10).tolist())
print("Unique participants:", df["CASEID"].nunique())
print("Unique visit codes:", df["VISIT"].nunique())
print("Duplicate participant-visit pairs:", duplicate_pairs)
print("Saved:", OUTPUT_FILE)
print("Original files were not modified.")
