
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

path = DATA / "macs_drugf1_selected_v2.csv"
df = pd.read_csv(
    path,
    dtype={"CASEID": "string", "VISIT": "string"},
    low_memory=False,
)

print("=== Source file shape ===")
print(df.shape)

print("\n=== Source columns ===")
print(df.columns.tolist())

reason_cols = [
    c for c in ["STOT1", "STOT2", "STOT3", "STOTH"]
    if c in df.columns
]

print("\n=== Stop-reason code counts ===")
for col in reason_cols:
    print(f"\n{col}:")
    print(df[col].astype("string").str.strip().value_counts(
        dropna=False
    ).head(20).to_string())

if "CASEID" in df.columns and "VISIT" in df.columns:
    for col in reason_cols:
        df[col] = df[col].astype("string").str.strip()

    df["_has_reason"] = df[reason_cols].apply(
        lambda row: row.notna()
        & ~row.isin(["", ".", "nan", "None"]),
        axis=0,
    ).any(axis=1)

    df["_has_code65"] = df[reason_cols].eq("65").any(axis=1)

    print("\n=== Example records containing code 65 ===")
    examples = df[df["_has_code65"]].head(30)

    show_cols = [
        c for c in [
            "CASEID", "VISIT", *reason_cols,
            "DATE", "DATE1", "YEAR", "LDATY"
        ] if c in df.columns
    ]
    print(examples[show_cols].to_string(index=False))

    print("\n=== Multiple medication records per participant-visit ===")
    counts = df.groupby(["CASEID", "VISIT"]).size()
    print(counts.describe().to_string())
    print("Participant-visits with multiple source records:",
          int((counts > 1).sum()))

    REPORTS.mkdir(parents=True, exist_ok=True)
    examples[show_cols].to_csv(
        REPORTS / "macs_code65_source_examples.csv",
        index=False,
    )
    print("\nSaved example records to reports/macs_code65_source_examples.csv")
