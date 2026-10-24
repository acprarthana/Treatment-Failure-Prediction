
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "interim" / "lab_rslt_clean.csv"

df = pd.read_csv(
    INPUT_FILE,
    dtype={"CASEID": str, "VISIT": str},
    keep_default_na=False,
)

viral_cols = ["VLOAD", "RO2VL", "R2SVL", "RTQ2VL", "RCOBVL"]
cd4_col = "LEU3N"

print("Dataset shape:", df.shape)
print("Unique participants:", df["CASEID"].nunique())
print("Unique visit codes:", df["VISIT"].nunique())

for col in viral_cols + [cd4_col]:
    if col not in df.columns:
        print(f"\n{col}: NOT FOUND")
        continue

    values = df[col].astype(str).str.strip()
    missing = values.isin(["", "."]).sum()
    numeric = pd.to_numeric(values.replace({".": pd.NA, "": pd.NA}),
                            errors="coerce")

    print(f"\n{'=' * 55}")
    print("Variable:", col)
    print("Missing:", missing, f"({missing / len(df) * 100:.2f}%)")
    print("Non-missing:", numeric.notna().sum())
    print("Distinct non-missing values:", numeric.nunique())

    print("Most common values:")
    print(numeric.value_counts(dropna=True).head(15).to_string())

    print("Numeric summary:")
    print(numeric.describe().to_string())

print("\nSample repeated measurements for three participants:")
for case_id in df["CASEID"].drop_duplicates().head(3):
    sample = df.loc[
        df["CASEID"] == case_id,
        ["CASEID", "VISIT", "LDATY", "LEU3N"] + viral_cols
    ]
    print(f"\nParticipant {case_id}")
    print(sample.head(12).to_string(index=False))

print("\nInspection only: no values or source files modified.")
