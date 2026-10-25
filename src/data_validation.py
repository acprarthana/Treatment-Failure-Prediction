
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

combined_path = DATA / "combined_ml_dataset_provisional.csv"
if not combined_path.exists():
    raise FileNotFoundError(f"Missing dataset: {combined_path}")

df = pd.read_csv(combined_path, low_memory=False)

# Structural validation
required = {
    "participant_id", "CASEID", "VISIT", "cohort",
    "CD4", "viral_load", "medication_records",
    "outcome", "outcome_status"
}
missing_columns = sorted(required - set(df.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

checks = [
    ("Total rows", len(df)),
    ("Unique participants", df["participant_id"].nunique()),
    ("Missing participant IDs", df["participant_id"].isna().sum()),
    ("Duplicate participant-visits",
     df.duplicated(["participant_id", "VISIT"]).sum()),
    ("Missing CD4", df["CD4"].isna().sum()),
    ("Missing viral load", df["viral_load"].isna().sum()),
    ("Missing medication match",
     df["medication_records"].isna().sum()),
    ("Missing outcome labels", df["outcome"].isna().sum()),
]

summary = pd.DataFrame(checks, columns=["Check", "Value"])
summary.to_csv(REPORTS / "member1_validation_summary.csv", index=False)

# Missingness by cohort and variable
variables = [
    c for c in ["CD4", "viral_load", "year", "medication_records",
                "reported_failure_signal", "outcome"]
    if c in df.columns
]
missingness = (
    df.groupby("cohort")[variables]
      .agg(lambda s: s.isna().sum())
      .T
      .rename_axis("Variable")
      .reset_index()
)
missingness.to_csv(REPORTS / "member1_missingness_by_cohort.csv",
                   index=False)

# Cohort-level summary
cohort_summary = df.groupby("cohort").agg(
    lab_visit_rows=("participant_id", "size"),
    unique_participants=("participant_id", "nunique"),
    visits_with_CD4=("CD4", "count"),
    visits_with_viral_load=("viral_load", "count"),
    visits_with_medication=("medication_records", "count"),
    visits_with_outcome=("outcome", "count"),
).reset_index()

cohort_summary.to_csv(REPORTS / "member1_cohort_summary.csv",
                      index=False)

# Outcome-status counts are descriptive only
status_counts = (
    df.groupby(["cohort", "outcome_status"], dropna=False)
      .size()
      .reset_index(name="rows")
)
status_counts.to_csv(REPORTS / "member1_outcome_status_audit.csv",
                     index=False)

report = f"""# Member 1: Data Cleaning and Validation Report

## Dataset
File: `data/interim/combined_ml_dataset_provisional.csv`

## Structural checks
{summary.to_markdown(index=False)}

## Cohort summary
{cohort_summary.to_markdown(index=False)}

## Important limitations
- This is a provisional combined lab-visit dataset.
- Medication information is aligned by cohort-specific participant ID and visit code.
- A matched visit does not guarantee that every medication or laboratory measurement is complete.
- Missing clinical values must be handled using a documented training-only strategy.
- Reported MACS failure-related medication-stop reasons are signals, not confirmed clinical failure outcomes.
- WIHS medication stop-reason semantics have not been verified.
- Cohort-derived sex assignments require source-document confirmation.
- Visit codes must not automatically be treated as calendar time.
- Treatment-failure outcome definition and prediction window remain unconfirmed.
- Do not train or report model performance until the target is clinically justified
  and predictors use only information available before the prediction time.

## Handoff
The dataset can be used for continued data-quality review and feature-engineering
development. Final outcome labels and modeling readiness require further validation.
"""
(REPORTS / "member1_validation_report.md").write_text(
    report, encoding="utf-8"
)

# Fail visibly if core structural requirements are violated
assert df["participant_id"].notna().all(), "Missing participant IDs found"
assert not df.duplicated(["participant_id", "VISIT"]).any(), \
    "Duplicate participant-visits found"

print("Member 1 report generation complete.")
print(f"Rows: {len(df):,}")
print(f"Participants: {df['participant_id'].nunique():,}")
print(f"Reports saved in: {REPORTS}")
print("Review member1_validation_report.md before handoff.")
