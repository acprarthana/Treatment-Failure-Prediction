import pandas as pd

p = "data/interim/hiv_longitudinal_outcomes_enriched_final_candidate.csv"
df = pd.read_csv(p, low_memory=False)

print("Medication/stop-reason-related columns:")
for c in df.columns:
    if any(k in c.lower() for k in ["stot", "stop", "medication", "art"]):
        print(c)

print("\nTarget labels by participant:")
labelled = df[df["treatment_outcome"].isin([0, 1])].copy()
per_person = labelled.groupby("participant_id").agg(
    labelled_rows=("treatment_outcome", "size"),
    distinct_labels=("treatment_outcome", "nunique"),
    label=("treatment_outcome", "first"),
    first_year=("year", "min"),
    last_year=("year", "max"),
)
print("Labelled rows:", len(labelled))
print("Participants:", len(per_person))
print("Participants with multiple labelled rows:",
      int((per_person["labelled_rows"] > 1).sum()))
print("Participants with both target labels:",
      int((per_person["distinct_labels"] > 1).sum()))
print("\nLabelled rows per participant:")
print(per_person["labelled_rows"].describe().to_string())

print("\nExample labelled rows:")
cols = [
    c for c in [
        "participant_id", "year", "VISIT", "DATE1yy",
        "CD4", "viral_load", "treatment_outcome",
        "treatment_outcome_status", "STOT1", "STOT2",
        "STOT3", "STOTH"
    ] if c in df.columns
]
print(labelled[cols].head(15).to_string(index=False))
