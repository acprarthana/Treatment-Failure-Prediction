
from pathlib import Path
import pandas as pd

path = Path("data/raw/WIHS/WIHS PDS/CSV/labsum.csv")

df = pd.read_csv(path, low_memory=False)

print("SHAPE:", df.shape)
print("\nDATA TYPES:")
print(df.dtypes)

print("\nUNIQUE PARTICIPANTS:", df["CASEID"].nunique())
print("UNIQUE VISITS:", df["VISIT"].nunique())

print("\nVISIT RANGE:")
print(df["VISIT"].min(), "to", df["VISIT"].max())

print("\nDUPLICATE PARTICIPANT-VISIT ROWS:",
      df.duplicated(["CASEID", "VISIT"]).sum())

print("\nMISSING VALUES:")
for col in ["VLOAD", "CD4N", "CD4P", "CD8N", "CD8P", "VLDT", "CD4DT"]:
    if col in df.columns:
        print(f"{col}: {df[col].isna().sum()} "
              f"({df[col].isna().mean() * 100:.2f}%)")

print("\nRECORDS PER PARTICIPANT:")
visits = df.groupby("CASEID")["VISIT"].nunique()
print(visits.describe())

print("\nVIRAL LOAD SUMMARY:")
print(df["VLOAD"].describe())

print("\nCD4 COUNT SUMMARY:")
print(df["CD4N"].describe())

print("\nMOST COMMON VISITS:")
print(df["VISIT"].value_counts().sort_index().to_string())
