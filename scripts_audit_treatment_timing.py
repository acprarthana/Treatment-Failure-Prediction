
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

p = DATA / "hiv_longitudinal_outcomes_enriched_final_candidate.csv"
df = pd.read_csv(p, low_memory=False)

# Keep only rows with a known proxy label.
labelled = df[df["treatment_outcome"].isin([0, 1])].copy()

print("=== 1. Label distribution by cohort ===")
print(pd.crosstab(
    labelled["cohort"],
    labelled["treatment_outcome"],
    margins=True
).to_string())

print("\n=== 2. Repeated labels per participant ===")
per_person = labelled.groupby("participant_id").agg(
    labelled_rows=("treatment_outcome", "size"),
    distinct_labels=("treatment_outcome", "nunique"),
    first_year=("year", "min"),
    last_year=("year", "max"),
).reset_index()

print("Participants with multiple labelled rows:",
      int((per_person["labelled_rows"] > 1).sum()))
print("Participants with both labels:",
      int((per_person["distinct_labels"] > 1).sum()))

print("\n=== 3. Label changes over time ===")
ordered = labelled.sort_values(
    ["participant_id", "year", "VISIT"]
).copy()

ordered["previous_label"] = (
    ordered.groupby("participant_id")["treatment_outcome"].shift()
)
ordered["label_changed"] = (
    ordered["previous_label"].notna()
    & ordered["treatment_outcome"].ne(ordered["previous_label"])
)

print("Adjacent labelled rows with different labels:",
      int(ordered["label_changed"].sum()))

print("\nExamples of participants with both labels:")
mixed_ids = per_person.loc[
    per_person["distinct_labels"] > 1, "participant_id"
].head(10)

cols = [
    c for c in [
        "participant_id", "year", "VISIT", "CD4", "viral_load",
        "treatment_outcome", "treatment_outcome_status",
        "treatment_outcome_source", "DATE1yy"
    ] if c in labelled.columns
]

print(
    ordered[ordered["participant_id"].isin(mixed_ids)][cols]
    .to_string(index=False)
)

print("\n=== 4. Check available time-related columns ===")
for c in df.columns:
    if any(k in c.lower() for k in [
        "date", "year", "visit", "event", "stop", "start"
    ]):
        print(c)

print("\n=== 5. Save audit tables ===")
REPORTS.mkdir(parents=True, exist_ok=True)
per_person.to_csv(
    REPORTS / "treatment_proxy_labels_per_participant_audit.csv",
    index=False
)
ordered[ordered["label_changed"]][cols].to_csv(
    REPORTS / "treatment_proxy_label_changes_audit.csv",
    index=False
)
print("Saved audit CSVs in:", REPORTS)
