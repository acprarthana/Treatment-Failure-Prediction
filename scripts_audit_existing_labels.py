
from pathlib import Path
import pandas as pd

source = Path(
    "data/interim/hiv_longitudinal_outcomes_enriched_final_candidate.csv"
)
report_dir = Path("reports")
report_dir.mkdir(parents=True, exist_ok=True)

cols = [
    "CASEID", "VISIT", "year", "cohort",
    "medication_record_count", "code65_record_count",
    "any_stop_reason_recorded", "reported_failure_signal",
    "treatment_outcome", "treatment_outcome_status",
    "treatment_outcome_source", "treatment_outcome_definition",
]

df = pd.read_csv(source, usecols=lambda c: c in cols, low_memory=False)

print("\n=== Dataset shape ===")
print(f"Rows: {len(df):,}")
print(f"Participants: {df['CASEID'].nunique():,}")

print("\n=== Target counts ===")
print(df["treatment_outcome"].value_counts(dropna=False).sort_index())

print("\n=== Target counts by cohort ===")
print(pd.crosstab(
    df["cohort"],
    df["treatment_outcome"].fillna("MISSING").astype(str),
    dropna=False
))

print("\n=== Relevant field availability ===")
for col in [
    "medication_record_count",
    "code65_record_count",
    "any_stop_reason_recorded",
    "reported_failure_signal",
    "treatment_outcome_status",
    "treatment_outcome_source",
]:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(15).to_string())

# Compare existing signals with the assigned binary target.
binary = df["treatment_outcome"].isin([0, 1])
for col in ["reported_failure_signal", "code65_record_count"]:
    if col in df.columns:
        print(f"\n=== {col} by assigned target ===")
        print(pd.crosstab(
            df.loc[binary, "treatment_outcome"],
            df.loc[binary, col].fillna("MISSING").astype(str),
            margins=True
        ))

# Save a compact audit table without changing the source dataset.
summary = {
    "rows": len(df),
    "participants": df["CASEID"].nunique(),
    "missing_target_rows": int(df["treatment_outcome"].isna().sum()),
    "label_0_rows": int((df["treatment_outcome"] == 0).sum()),
    "label_1_rows": int((df["treatment_outcome"] == 1).sum()),
    "participants_with_multiple_labels": int(
        (df.groupby("CASEID")["treatment_outcome"].nunique() > 1).sum()
    ),
}
pd.Series(summary, name="value").to_csv(
    report_dir / "existing_label_audit_summary.csv"
)
print("\nSaved reports/existing_label_audit_summary.csv")
