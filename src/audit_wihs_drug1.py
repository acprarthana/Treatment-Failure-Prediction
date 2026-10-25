
from pathlib import Path
import pandas as pd

path = Path("data/raw/WIHS/WIHS PDS/CSV/drug1.csv")
df = pd.read_csv(path, encoding="cp1252", low_memory=False)

print("SHAPE:", df.shape)
print("\nUNIQUE PARTICIPANTS:", df["CASEID"].nunique())
print("UNIQUE VISITS:", df["VISIT"].nunique())
print("VISIT RANGE:", df["VISIT"].min(), "to", df["VISIT"].max())

print("\nMISSING VALUES (%):")
cols = [
    "CASEID", "VISIT", "VTYP", "DRUGD1",
    "BGNMD1", "BGNYD1", "CURRD1", "INITD1",
    "DCMPD1", "MREPD1", "SOMRD1", "ESTAT"
]
for col in cols:
    if col in df.columns:
        print(f"{col}: {df[col].isna().mean() * 100:.2f}%")

print("\nDUPLICATE PARTICIPANT-VISIT ROWS:",
      df.duplicated(["CASEID", "VISIT"]).sum())

print("\nROWS PER PARTICIPANT:")
print(df.groupby("CASEID").size().describe())

print("\nCOMMON DRUG CODES:")
print(df["DRUGD1"].value_counts(dropna=False).head(20).to_string())

print("\nCURRENT-STATUS CODES:")
print(df["CURRD1"].value_counts(dropna=False).head(20).to_string())

print("\nVISIT TYPE CODES:")
print(df["VTYP"].value_counts(dropna=False).to_string())
