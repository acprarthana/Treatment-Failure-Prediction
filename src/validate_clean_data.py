
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"

df = pd.read_csv(FILE, dtype={"CASEID": "string", "VISIT": "string"})

print("SHAPE")
print(df.shape)

print("\nPARTICIPANTS AND VISITS")
print("Unique participants:", df["CASEID"].nunique())
print("Unique visit codes:", df["VISIT"].nunique())
print("Duplicate participant-visit pairs:",
      df.duplicated(["CASEID", "VISIT"]).sum())

print("\nVISIT CODE SAMPLE")
print(sorted(df["VISIT"].dropna().unique())[:15])

print("\nYEAR RANGE")
print("Minimum year:", df["LDATY"].min())
print("Maximum year:", df["LDATY"].max())

print("\nMEASUREMENT SUMMARIES")
for col in ["LEU3N", "LEU2N", "LEU4N"]:
    values = pd.to_numeric(df[col], errors="coerce")
    print(f"\n{col}")
    print(values.describe().to_string())
    print("Negative values:", (values < 0).sum())

print("\nVIRAL-LOAD RAW VALUE CHECK")
for col in ["VLOAD", "R2SVL", "RTQ2VL"]:
    values = df[col].dropna().astype(str).str.strip()
    numeric = pd.to_numeric(values, errors="coerce")
    print(f"\n{col}")
    print("Non-numeric nonmissing values:", numeric.isna().sum())
    print("Minimum numeric code:", numeric.min())
    print("Maximum numeric code:", numeric.max())
    print("Most common values:")
    print(values.value_counts().head(8).to_string())

print("\nValidation checks completed.")
