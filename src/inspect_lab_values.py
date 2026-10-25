
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "interim" / "lab_rslt.csv"

df = pd.read_csv(
    CSV_FILE,
    dtype=str,
    keep_default_na=False,
)

columns = ["CASEID", "VISIT", "LDATY", "LEU3N", "LEU2N",
           "LEU4N", "VLOAD", "R2SVL", "RTQ2VL"]

print("Dataset shape:", df.shape)

print("\nMissing-marker counts:")
for col in columns:
    if col in df.columns:
        values = df[col].str.strip()
        print(
            f"{col}: blank={(values == '').sum():,}, "
            f"period={(values == '.').sum():,}"
        )

print("\nMost common raw values:")
for col in ["LEU3N", "LEU2N", "LEU4N", "VLOAD", "R2SVL", "RTQ2VL"]:
    print(f"\n{col}")
    print(df[col].str.strip().value_counts().head(12).to_string())

print("\nSample records with both CD4 and viral-load values:")
sample = df[
    (df["LEU3N"].str.strip().ne("")) &
    (df["LEU3N"].str.strip().ne(".")) &
    (df["VLOAD"].str.strip().ne("")) &
    (df["VLOAD"].str.strip().ne("."))
]

print(sample[columns].head(10).to_string(index=False))
