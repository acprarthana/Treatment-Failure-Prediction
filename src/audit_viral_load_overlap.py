
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"

df = pd.read_csv(
    INPUT_FILE,
    dtype={"CASEID": str, "VISIT": str},
    keep_default_na=False,
)

assays = ["VLOAD", "RO2VL", "R2SVL", "RTQ2VL", "RCOBVL"]
assays = [c for c in assays if c in df.columns]

for col in assays:
    df[col] = pd.to_numeric(
        df[col].replace({"": pd.NA, ".": pd.NA}),
        errors="coerce"
    )

# 1. How many assays are populated per participant-visit row?
df["assay_count"] = df[assays].notna().sum(axis=1)

print("Rows by number of populated viral-load fields:")
print(df["assay_count"].value_counts().sort_index().to_string())

print("\nAssay availability combinations:")
patterns = (
    df[assays].notna()
    .apply(lambda row: "+".join(row.index[row].tolist()) or "NONE", axis=1)
)
print(patterns.value_counts().head(20).to_string())

# 2. Compare values only where two or more assays are populated.
multi = df.loc[df["assay_count"] >= 2, ["CASEID", "VISIT", "LDATY"] + assays].copy()
print("\nRows with two or more populated assays:", len(multi))

if len(multi):
    def row_unique_count(row):
        vals = row[assays].dropna().tolist()
        return len(set(vals))

    multi["distinct_value_count"] = multi.apply(row_unique_count, axis=1)
    print("Same value across all populated assays:",
          (multi["distinct_value_count"] == 1).sum())
    print("Different values across populated assays:",
          (multi["distinct_value_count"] > 1).sum())

    print("\nExamples where populated assay values differ:")
    print(
        multi.loc[multi["distinct_value_count"] > 1]
        .head(15)
        .to_string(index=False)
    )

# 3. Per-assay longitudinal availability.
print("\nLongitudinal availability by assay:")
for col in assays:
    observed = df.loc[df[col].notna(), ["CASEID", "LDATY", col]].copy()
    observed["LDATY"] = pd.to_numeric(observed["LDATY"], errors="coerce")
    per_person = observed.groupby("CASEID").agg(
        observations=(col, "count"),
        distinct_years=("LDATY", "nunique"),
    )

    print(f"\n{col}")
    print("Rows with a value:", len(observed))
    print("Participants with >=1 value:", len(per_person))
    print("Participants with >=2 values:", (per_person["observations"] >= 2).sum())
    print("Participants with >=3 values:", (per_person["observations"] >= 3).sum())
    print("Participants with >=2 distinct years:",
          (per_person["distinct_years"] >= 2).sum())

print("\nAudit only. No data files were modified.")
