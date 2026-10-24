
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "interim" / "lab_rslt.csv"
OUTPUT_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False,
)

# Preserve IDs and visit codes as strings.
# Normalize whitespace and documented raw missing markers.
for col in df.columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace(".", pd.NA)

# Convert only fields verified as integer-valued measurements.
for col in ["LDATY", "LEU3N", "LEU2N", "LEU4N"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

assert len(df) == 143263
assert df["CASEID"].notna().all()
assert df["CASEID"].str.len().eq(5).all()
assert df["VISIT"].notna().all()

df.to_csv(OUTPUT_FILE, index=False)

print("Cleaned shape:", df.shape)
print("Example participant IDs:", df["CASEID"].head().tolist())
print("Example visit codes:", df["VISIT"].head().tolist())
print("Unique participants:", df["CASEID"].nunique())
print("Duplicate participant-visit pairs:",
      df.duplicated(["CASEID", "VISIT"]).sum())
print("Saved:", OUTPUT_FILE)
