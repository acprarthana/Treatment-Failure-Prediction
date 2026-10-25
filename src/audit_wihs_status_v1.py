
from pathlib import Path
import pandas as pd

root = Path("data/raw/WIHS/WIHS PDS/CSV")

labs = pd.read_csv(root / "labsum.csv", low_memory=False)
status = pd.read_csv(root / "hivstat.csv", low_memory=False)

print("=== HIV STATUS ===")
print(status["STATUS"].value_counts(dropna=False).sort_index())
print("Unique participants:", status["CASEID"].nunique())
print("Duplicate status IDs:", status["CASEID"].duplicated().sum())

print("\n=== LAB DATA: VIRAL-LOAD DETECTION STATUS ===")
print(labs["UD_VL"].value_counts(dropna=False).sort_index())

print("\n=== LAB DATA: LOWER QUANTIFICATION LIMIT ===")
print(labs["LLQ_VL"].value_counts(dropna=False).sort_index())

print("\n=== VIRAL LOAD BY DETECTION STATUS ===")
print(
    labs.groupby("UD_VL", dropna=False)["VLOAD"]
    .agg(["count", "min", "median", "max"])
    .to_string()
)

print("\n=== HIV STATUS AND LAB RECORD OVERLAP ===")
lab_ids = set(labs["CASEID"].dropna())
status_ids = set(status["CASEID"].dropna())

print("Lab participants:", len(lab_ids))
print("Status participants:", len(status_ids))
print("Participants in both files:", len(lab_ids & status_ids))
print("Lab participants without status record:", len(lab_ids - status_ids))

print("\n=== STATUS COUNTS FOR LAB PARTICIPANTS ===")
merged = labs[["CASEID"]].drop_duplicates().merge(
    status[["CASEID", "STATUS"]],
    on="CASEID",
    how="left",
    validate="one_to_one"
)
print(merged["STATUS"].value_counts(dropna=False).sort_index())
