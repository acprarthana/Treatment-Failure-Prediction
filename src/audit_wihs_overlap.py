
from pathlib import Path
import pandas as pd

root = Path("data/raw/WIHS/WIHS PDS/CSV")

labs = pd.read_csv(root / "labsum.csv", low_memory=False)
status = pd.read_csv(root / "hivstat.csv", low_memory=False)
drugs = pd.read_csv(
    root / "drug1.csv", encoding="cp1252", low_memory=False
)
outcomes = pd.read_csv(root / "outcome.csv", low_memory=False)

# Start with participants classified as HIV-positive at baseline
# according to the WIHS status summary.
positive_ids = set(
    status.loc[status["STATUS"] == 2, "CASEID"]
)

lab_ids = set(labs["CASEID"])
drug_ids = set(drugs["CASEID"])
outcome_ids = set(outcomes["CASEID"])

print("=== PARTICIPANT OVERLAP ===")
print("HIV-positive status-2 participants:", len(positive_ids))
print("Status-2 participants with labs:", len(positive_ids & lab_ids))
print("Status-2 participants with ART records:",
      len(positive_ids & drug_ids))
print("Status-2 participants with outcome records:",
      len(positive_ids & outcome_ids))
print("Status-2 participants with both labs and ART records:",
      len(positive_ids & lab_ids & drug_ids))

# Restrict the medication records to the status-2 group.
positive_drugs = drugs[drugs["CASEID"].isin(positive_ids)].copy()

print("\n=== ART RECORDS FOR STATUS-2 PARTICIPANTS ===")
print("Medication rows:", len(positive_drugs))
print("Participants:", positive_drugs["CASEID"].nunique())
print("\nCurrently taking drug:")
print(positive_drugs["CURRD1"].value_counts(dropna=False).sort_index())

print("\nTreatment-start year values:")
print(positive_drugs["BGNYD1"].value_counts(dropna=False).sort_index().to_string())

print("\n=== OUTCOME FILE ===")
print("Outcome rows:", len(outcomes))
print("Participants:", outcomes["CASEID"].nunique())

for col in ["OUTCAIDS", "OUTCDTH", "OUTCCANC",
            "OUTCTB", "EVNTSTAT", "EVNTCODE"]:
    if col in outcomes.columns:
        print(f"\n{col}:")
        print(outcomes[col].value_counts(dropna=False).sort_index().to_string())

print("\n=== OUTCOME EVENT STATUS BY AIDS FLAG ===")
if {"OUTCAIDS", "EVNTSTAT"}.issubset(outcomes.columns):
    print(pd.crosstab(
        outcomes["OUTCAIDS"],
        outcomes["EVNTSTAT"],
        dropna=False
    ))
